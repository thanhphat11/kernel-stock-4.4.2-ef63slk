/*
 *  BLUR, or complete fairness queueing, disk scheduler.
 *
 *  Based on ideas from a previously unfinished io
 *  scheduler (round robin per-process disk scheduling) and Andrea Arcangeli.
 *
 *  Copyright (C) 2003 Jens Axboe <axboe@kernel.dk>
 */
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/blkdev.h>
#include <linux/elevator.h>
#include <linux/jiffies.h>
#include <linux/rbtree.h>
#include <linux/ioprio.h>
#include <linux/blktrace_api.h>
#include "blk.h"
#include "blur.h"

/*
 * tunables
 */
/* max queue in one round of service */
static const int blur_quantum = 8;
static const int blur_fifo_expire[2] = { HZ / 4, HZ / 8 };
/* maximum backwards seek, in KiB */
static const int blur_back_max = 16 * 1024;
/* penalty of a backwards seek */
static const int blur_back_penalty = 2;
static const int blur_slice_sync = HZ / 10;
static int blur_slice_async = HZ / 25;
static const int blur_slice_async_rq = 2;
static int blur_slice_idle = HZ / 125;
static int blur_group_idle = HZ / 125;
static const int blur_target_latency = HZ * 3/10; /* 300 ms */
static const int blur_hist_divisor = 4;

/*
 * offset from end of service tree
 */
#define blur_IDLE_DELAY		(HZ / 5)

/*
 * below this threshold, we consider thinktime immediate
 */
#define blur_MIN_TT		(2)

#define blur_SLICE_SCALE		(5)
#define blur_HW_QUEUE_MIN	(5)
#define blur_SERVICE_SHIFT       12

#define blurQ_SEEK_THR		(sector_t)(8 * 100)
#define blurQ_CLOSE_THR		(sector_t)(8 * 1024)
#define blurQ_SECT_THR_NONROT	(sector_t)(2 * 32)
#define blurQ_SEEKY(blurq)	(hweight32(blurq->seek_history) > 32/8)

#define RQ_CIC(rq)		icq_to_cic((rq)->elv.icq)
#define RQ_blurQ(rq)		(struct blur_queue *) ((rq)->elv.priv[0])
#define RQ_blurG(rq)		(struct blur_group *) ((rq)->elv.priv[1])

static struct kmem_cache *blur_pool;

#define blur_PRIO_LISTS		IOPRIO_BE_NR
#define blur_class_idle(blurq)	((blurq)->ioprio_class == IOPRIO_CLASS_IDLE)
#define blur_class_rt(blurq)	((blurq)->ioprio_class == IOPRIO_CLASS_RT)

#define sample_valid(samples)	((samples) > 80)
#define rb_entry_blurg(node)	rb_entry((node), struct blur_group, rb_node)

struct blur_ttime {
	unsigned long last_end_request;

	unsigned long ttime_total;
	unsigned long ttime_samples;
	unsigned long ttime_mean;
};

/*
 * Most of our rbtree usage is for sorting with min extraction, so
 * if we cache the leftmost node we don't have to walk down the tree
 * to find it. Idea borrowed from Ingo Molnars CFS scheduler. We should
 * move this into the elevator for the rq sorting as well.
 */
struct blur_rb_root {
	struct rb_root rb;
	struct rb_node *left;
	unsigned count;
	unsigned total_weight;
	u64 min_vdisktime;
	struct blur_ttime ttime;
};
#define blur_RB_ROOT	(struct blur_rb_root) { .rb = RB_ROOT, \
			.ttime = {.last_end_request = jiffies,},}

/*
 * Per process-grouping structure
 */
struct blur_queue {
	/* reference count */
	int ref;
	/* various state flags, see below */
	unsigned int flags;
	/* parent blur_data */
	struct blur_data *blurd;
	/* service_tree member */
	struct rb_node rb_node;
	/* service_tree key */
	unsigned long rb_key;
	/* prio tree member */
	struct rb_node p_node;
	/* prio tree root we belong to, if any */
	struct rb_root *p_root;
	/* sorted list of pending requests */
	struct rb_root sort_list;
	/* if fifo isn't expired, next request to serve */
	struct request *next_rq;
	/* requests queued in sort_list */
	int queued[2];
	/* currently allocated requests */
	int allocated[2];
	/* fifo list of requests in sort_list */
	struct list_head fifo;

	/* time when queue got scheduled in to dispatch first request. */
	unsigned long dispatch_start;
	unsigned int allocated_slice;
	unsigned int slice_dispatch;
	/* time when first request from queue completed and slice started. */
	unsigned long slice_start;
	unsigned long slice_end;
	long slice_resid;

	/* pending priority requests */
	int prio_pending;
	/* number of requests that are on the dispatch list or inside driver */
	int dispatched;

	/* io prio of this group */
	unsigned short ioprio, org_ioprio;
	unsigned short ioprio_class;

	pid_t pid;

	u32 seek_history;
	sector_t last_request_pos;

	struct blur_rb_root *service_tree;
	struct blur_queue *new_blurq;
	struct blur_group *blurg;
	/* Number of sectors dispatched from queue in single dispatch round */
	unsigned long nr_sectors;
};

/*
 * First index in the service_trees.
 * IDLE is handled separately, so it has negative index
 */
enum wl_prio_t {
	BE_WORKLOAD = 0,
	RT_WORKLOAD = 1,
	IDLE_WORKLOAD = 2,
	blur_PRIO_NR,
};

/*
 * Second index in the service_trees.
 */
enum wl_type_t {
	ASYNC_WORKLOAD = 0,
	SYNC_NOIDLE_WORKLOAD = 1,
	SYNC_WORKLOAD = 2
};

/* This is per cgroup per device grouping structure */
struct blur_group {
	/* group service_tree member */
	struct rb_node rb_node;

	/* group service_tree key */
	u64 vdisktime;
	unsigned int weight;
	unsigned int new_weight;
	bool needs_update;

	/* number of blurq currently on this group */
	int nr_blurq;

	/*
	 * Per group busy queues average. Useful for workload slice calc. We
	 * create the array for each prio class but at run time it is used
	 * only for RT and BE class and slot for IDLE class remains unused.
	 * This is primarily done to avoid confusion and a gcc warning.
	 */
	unsigned int busy_queues_avg[blur_PRIO_NR];
	/*
	 * rr lists of queues with requests. We maintain service trees for
	 * RT and BE classes. These trees are subdivided in subclasses
	 * of SYNC, SYNC_NOIDLE and ASYNC based on workload type. For IDLE
	 * class there is no subclassification and all the blur queues go on
	 * a single tree service_tree_idle.
	 * Counts are embedded in the blur_rb_root
	 */
	struct blur_rb_root service_trees[2][3];
	struct blur_rb_root service_tree_idle;

	unsigned long saved_workload_slice;
	enum wl_type_t saved_workload;
	enum wl_prio_t saved_serving_prio;
	struct blkio_group blkg;
#ifdef CONFIG_BLUR_GROUP_IOSCHED
	struct hlist_node blurd_node;
	int ref;
#endif
	/* number of requests that are on the dispatch list or inside driver */
	int dispatched;
	struct blur_ttime ttime;
};

struct blur_io_cq {
	struct io_cq		icq;		/* must be the first member */
	struct blur_queue	*blurq[2];
	struct blur_ttime	ttime;
};

/*
 * Per block device queue structure
 */
struct blur_data {
	struct request_queue *queue;
	/* Root service tree for blur_groups */
	struct blur_rb_root grp_service_tree;
	struct blur_group root_group;

	/*
	 * The priority currently being served
	 */
	enum wl_prio_t serving_prio;
	enum wl_type_t serving_type;
	unsigned long workload_expires;
	struct blur_group *serving_group;

	/*
	 * Each priority tree is sorted by next_request position.  These
	 * trees are used when determining if two or more queues are
	 * interleaving requests (see blur_close_cooperator).
	 */
	struct rb_root prio_trees[blur_PRIO_LISTS];

	unsigned int busy_queues;
	unsigned int busy_sync_queues;

	int rq_in_driver;
	int rq_in_flight[2];

	/*
	 * queue-depth detection
	 */
	int rq_queued;
	int hw_tag;
	/*
	 * hw_tag can be
	 * -1 => indeterminate, (blur will behave as if NCQ is present, to allow better detection)
	 *  1 => NCQ is present (hw_tag_est_depth is the estimated max depth)
	 *  0 => no NCQ
	 */
	int hw_tag_est_depth;
	unsigned int hw_tag_samples;

	/*
	 * idle window management
	 */
	struct timer_list idle_slice_timer;
	struct work_struct unplug_work;

	struct blur_queue *active_queue;
	struct blur_io_cq *active_cic;

	/*
	 * async queue for each priority case
	 */
	struct blur_queue *async_blurq[2][IOPRIO_BE_NR];
	struct blur_queue *async_idle_blurq;

	sector_t last_position;

	/*
	 * tunables, see top of file
	 */
	unsigned int blur_quantum;
	unsigned int blur_fifo_expire[2];
	unsigned int blur_back_penalty;
	unsigned int blur_back_max;
	unsigned int blur_slice[2];
	unsigned int blur_slice_async_rq;
	unsigned int blur_slice_idle;
	unsigned int blur_group_idle;
	unsigned int blur_latency;
	unsigned int blur_target_latency;

	/*
	 * Fallback dummy blurq for extreme OOM conditions
	 */
	struct blur_queue oom_blurq;

	unsigned long last_delayed_sync;

	/* List of blur groups being managed on this device*/
	struct hlist_head blurg_list;

	/* Number of groups which are on blkcg->blkg_list */
	unsigned int nr_blkcg_linked_grps;
};

static struct blur_group *blur_get_next_blurg(struct blur_data *blurd);

static struct blur_rb_root *service_tree_for(struct blur_group *blurg,
					    enum wl_prio_t prio,
					    enum wl_type_t type)
{
	if (!blurg)
		return NULL;

	if (prio == IDLE_WORKLOAD)
		return &blurg->service_tree_idle;

	return &blurg->service_trees[prio][type];
}

enum blurq_state_flags {
	blur_blurQ_FLAG_on_rr = 0,	/* on round-robin busy list */
	blur_blurQ_FLAG_wait_request,	/* waiting for a request */
	blur_blurQ_FLAG_must_dispatch,	/* must be allowed a dispatch */
	blur_blurQ_FLAG_must_alloc_slice,	/* per-slice must_alloc flag */
	blur_blurQ_FLAG_fifo_expire,	/* FIFO checked in this slice */
	blur_blurQ_FLAG_idle_window,	/* slice idling enabled */
	blur_blurQ_FLAG_prio_changed,	/* task priority has changed */
	blur_blurQ_FLAG_slice_new,	/* no requests dispatched in slice */
	blur_blurQ_FLAG_sync,		/* synchronous queue */
	blur_blurQ_FLAG_coop,		/* blurq is shared */
	blur_blurQ_FLAG_split_coop,	/* shared blurq will be splitted */
	blur_blurQ_FLAG_deep,		/* sync blurq experienced large depth */
	blur_blurQ_FLAG_wait_busy,	/* Waiting for next request */
};

#define blur_blurQ_FNS(name)						\
static inline void blur_mark_blurq_##name(struct blur_queue *blurq)		\
{									\
	(blurq)->flags |= (1 << blur_blurQ_FLAG_##name);			\
}									\
static inline void blur_clear_blurq_##name(struct blur_queue *blurq)	\
{									\
	(blurq)->flags &= ~(1 << blur_blurQ_FLAG_##name);			\
}									\
static inline int blur_blurq_##name(const struct blur_queue *blurq)		\
{									\
	return ((blurq)->flags & (1 << blur_blurQ_FLAG_##name)) != 0;	\
}

blur_blurQ_FNS(on_rr);
blur_blurQ_FNS(wait_request);
blur_blurQ_FNS(must_dispatch);
blur_blurQ_FNS(must_alloc_slice);
blur_blurQ_FNS(fifo_expire);
blur_blurQ_FNS(idle_window);
blur_blurQ_FNS(prio_changed);
blur_blurQ_FNS(slice_new);
blur_blurQ_FNS(sync);
blur_blurQ_FNS(coop);
blur_blurQ_FNS(split_coop);
blur_blurQ_FNS(deep);
blur_blurQ_FNS(wait_busy);
#undef blur_blurQ_FNS

#ifdef CONFIG_blur_GROUP_IOSCHED
#define blur_log_blurq(blurd, blurq, fmt, args...)	\
	blk_add_trace_msg((blurd)->queue, "blur%d%c %s " fmt, (blurq)->pid, \
			blur_blurq_sync((blurq)) ? 'S' : 'A', \
			blkg_path(&(blurq)->blurg->blkg), ##args)

#define blur_log_blurg(blurd, blurg, fmt, args...)				\
	blk_add_trace_msg((blurd)->queue, "%s " fmt,			\
				blkg_path(&(blurg)->blkg), ##args)       \

#else
#define blur_log_blurq(blurd, blurq, fmt, args...)	\
	blk_add_trace_msg((blurd)->queue, "blur%d " fmt, (blurq)->pid, ##args)
#define blur_log_blurg(blurd, blurg, fmt, args...)		do {} while (0)
#endif
#define blur_log(blurd, fmt, args...)	\
	blk_add_trace_msg((blurd)->queue, "blur " fmt, ##args)

/* Traverses through blur group service trees */
#define for_each_blurg_st(blurg, i, j, st) \
	for (i = 0; i <= IDLE_WORKLOAD; i++) \
		for (j = 0, st = i < IDLE_WORKLOAD ? &blurg->service_trees[i][j]\
			: &blurg->service_tree_idle; \
			(i < IDLE_WORKLOAD && j <= SYNC_WORKLOAD) || \
			(i == IDLE_WORKLOAD && j == 0); \
			j++, st = i < IDLE_WORKLOAD ? \
			&blurg->service_trees[i][j]: NULL) \

static inline bool blur_io_thinktime_big(struct blur_data *blurd,
	struct blur_ttime *ttime, bool group_idle)
{
	unsigned long slice;
	if (!sample_valid(ttime->ttime_samples))
		return false;
	if (group_idle)
		slice = blurd->blur_group_idle;
	else
		slice = blurd->blur_slice_idle;
	return ttime->ttime_mean > slice;
}

static inline bool iops_mode(struct blur_data *blurd)
{
	/*
	 * If we are not idling on queues and it is a NCQ drive, parallel
	 * execution of requests is on and measuring time is not possible
	 * in most of the cases until and unless we drive shallower queue
	 * depths and that becomes a performance bottleneck. In such cases
	 * switch to start providing fairness in terms of number of IOs.
	 */
	if (!blurd->blur_slice_idle && blurd->hw_tag)
		return true;
	else
		return false;
}

static inline enum wl_prio_t blurq_prio(struct blur_queue *blurq)
{
	if (blur_class_idle(blurq))
		return IDLE_WORKLOAD;
	if (blur_class_rt(blurq))
		return RT_WORKLOAD;
	return BE_WORKLOAD;
}


static enum wl_type_t blurq_type(struct blur_queue *blurq)
{
	if (!blur_blurq_sync(blurq))
		return ASYNC_WORKLOAD;
	if (!blur_blurq_idle_window(blurq))
		return SYNC_NOIDLE_WORKLOAD;
	return SYNC_WORKLOAD;
}

static inline int blur_group_busy_queues_wl(enum wl_prio_t wl,
					struct blur_data *blurd,
					struct blur_group *blurg)
{
	if (wl == IDLE_WORKLOAD)
		return blurg->service_tree_idle.count;

	return blurg->service_trees[wl][ASYNC_WORKLOAD].count
		+ blurg->service_trees[wl][SYNC_NOIDLE_WORKLOAD].count
		+ blurg->service_trees[wl][SYNC_WORKLOAD].count;
}

static inline int blurg_busy_async_queues(struct blur_data *blurd,
					struct blur_group *blurg)
{
	return blurg->service_trees[RT_WORKLOAD][ASYNC_WORKLOAD].count
		+ blurg->service_trees[BE_WORKLOAD][ASYNC_WORKLOAD].count;
}

static void blur_dispatch_insert(struct request_queue *, struct request *);
static struct blur_queue *blur_get_queue(struct blur_data *, bool,
				       struct io_context *, gfp_t);

static inline struct blur_io_cq *icq_to_cic(struct io_cq *icq)
{
	/* cic->icq is the first member, %NULL will convert to %NULL */
	return container_of(icq, struct blur_io_cq, icq);
}

static inline struct blur_io_cq *blur_cic_lookup(struct blur_data *blurd,
					       struct io_context *ioc)
{
	if (ioc)
		return icq_to_cic(ioc_lookup_icq(ioc, blurd->queue));
	return NULL;
}

static inline struct blur_queue *cic_to_blurq(struct blur_io_cq *cic, bool is_sync)
{
	return cic->blurq[is_sync];
}

static inline void cic_set_blurq(struct blur_io_cq *cic, struct blur_queue *blurq,
				bool is_sync)
{
	cic->blurq[is_sync] = blurq;
}

static inline struct blur_data *cic_to_blurd(struct blur_io_cq *cic)
{
	return cic->icq.q->elevator->elevator_data;
}

/*
 * We regard a request as SYNC, if it's either a read or has the SYNC bit
 * set (in which case it could also be direct WRITE).
 */
static inline bool blur_bio_sync(struct bio *bio)
{
	return bio_data_dir(bio) == READ || (bio->bi_rw & REQ_SYNC);
}

/*
 * scheduler run of queue, if there are requests pending and no one in the
 * driver that will restart queueing
 */
static inline void blur_schedule_dispatch(struct blur_data *blurd)
{
	if (blurd->busy_queues) {
		blur_log(blurd, "schedule dispatch");
		kblockd_schedule_work(blurd->queue, &blurd->unplug_work);
	}
}

/*
 * Scale schedule slice based on io priority. Use the sync time slice only
 * if a queue is marked sync and has sync io queued. A sync queue with async
 * io only, should not get full sync slice length.
 */
static inline int blur_prio_slice(struct blur_data *blurd, bool sync,
				 unsigned short prio)
{
	const int base_slice = blurd->blur_slice[sync];

	WARN_ON(prio >= IOPRIO_BE_NR);

	return base_slice + (base_slice/blur_SLICE_SCALE * (4 - prio));
}

static inline int
blur_prio_to_slice(struct blur_data *blurd, struct blur_queue *blurq)
{
	return blur_prio_slice(blurd, blur_blurq_sync(blurq), blurq->ioprio);
}

static inline u64 blur_scale_slice(unsigned long delta, struct blur_group *blurg)
{
	u64 d = delta << blur_SERVICE_SHIFT;

	d = d * BLKIO_WEIGHT_DEFAULT;
	do_div(d, blurg->weight);
	return d;
}

static inline u64 max_vdisktime(u64 min_vdisktime, u64 vdisktime)
{
	s64 delta = (s64)(vdisktime - min_vdisktime);
	if (delta > 0)
		min_vdisktime = vdisktime;

	return min_vdisktime;
}

static inline u64 min_vdisktime(u64 min_vdisktime, u64 vdisktime)
{
	s64 delta = (s64)(vdisktime - min_vdisktime);
	if (delta < 0)
		min_vdisktime = vdisktime;

	return min_vdisktime;
}

static void update_min_vdisktime(struct blur_rb_root *st)
{
	struct blur_group *blurg;

	if (st->left) {
		blurg = rb_entry_blurg(st->left);
		st->min_vdisktime = max_vdisktime(st->min_vdisktime,
						  blurg->vdisktime);
	}
}

/*
 * get averaged number of queues of RT/BE priority.
 * average is updated, with a formula that gives more weight to higher numbers,
 * to quickly follows sudden increases and decrease slowly
 */

static inline unsigned blur_group_get_avg_queues(struct blur_data *blurd,
					struct blur_group *blurg, bool rt)
{
	unsigned min_q, max_q;
	unsigned mult  = blur_hist_divisor - 1;
	unsigned round = blur_hist_divisor / 2;
	unsigned busy = blur_group_busy_queues_wl(rt, blurd, blurg);

	min_q = min(blurg->busy_queues_avg[rt], busy);
	max_q = max(blurg->busy_queues_avg[rt], busy);
	blurg->busy_queues_avg[rt] = (mult * max_q + min_q + round) /
		blur_hist_divisor;
	return blurg->busy_queues_avg[rt];
}

static inline unsigned
blur_group_slice(struct blur_data *blurd, struct blur_group *blurg)
{
	struct blur_rb_root *st = &blurd->grp_service_tree;

	return blurd->blur_target_latency * blurg->weight / st->total_weight;
}

static inline unsigned
blur_scaled_blurq_slice(struct blur_data *blurd, struct blur_queue *blurq)
{
	unsigned slice = blur_prio_to_slice(blurd, blurq);
	if (blurd->blur_latency) {
		/*
		 * interested queues (we consider only the ones with the same
		 * priority class in the blur group)
		 */
		unsigned iq = blur_group_get_avg_queues(blurd, blurq->blurg,
						blur_class_rt(blurq));
		unsigned sync_slice = blurd->blur_slice[1];
		unsigned expect_latency = sync_slice * iq;
		unsigned group_slice = blur_group_slice(blurd, blurq->blurg);

		if (expect_latency > group_slice) {
			unsigned base_low_slice = 2 * blurd->blur_slice_idle;
			/* scale low_slice according to IO priority
			 * and sync vs async */
			unsigned low_slice =
				min(slice, base_low_slice * slice / sync_slice);
			/* the adapted slice value is scaled to fit all iqs
			 * into the target latency */
			slice = max(slice * group_slice / expect_latency,
				    low_slice);
		}
	}
	return slice;
}

static inline void
blur_set_prio_slice(struct blur_data *blurd, struct blur_queue *blurq)
{
	unsigned slice = blur_scaled_blurq_slice(blurd, blurq);

	blurq->slice_start = jiffies;
	blurq->slice_end = jiffies + slice;
	blurq->allocated_slice = slice;
	blur_log_blurq(blurd, blurq, "set_slice=%lu", blurq->slice_end - jiffies);
}

/*
 * We need to wrap this check in blur_blurq_slice_new(), since ->slice_end
 * isn't valid until the first request from the dispatch is activated
 * and the slice time set.
 */
static inline bool blur_slice_used(struct blur_queue *blurq)
{
	if (blur_blurq_slice_new(blurq))
		return false;
	if (time_before(jiffies, blurq->slice_end))
		return false;

	return true;
}

/*
 * Lifted from AS - choose which of rq1 and rq2 that is best served now.
 * We choose the request that is closest to the head right now. Distance
 * behind the head is penalized and only allowed to a certain extent.
 */
static struct request *
blur_choose_req(struct blur_data *blurd, struct request *rq1, struct request *rq2, sector_t last)
{
	sector_t s1, s2, d1 = 0, d2 = 0;
	unsigned long back_max;
#define blur_RQ1_WRAP	0x01 /* request 1 wraps */
#define blur_RQ2_WRAP	0x02 /* request 2 wraps */
	unsigned wrap = 0; /* bit mask: requests behind the disk head? */

	if (rq1 == NULL || rq1 == rq2)
		return rq2;
	if (rq2 == NULL)
		return rq1;

	if (rq_is_sync(rq1) != rq_is_sync(rq2))
		return rq_is_sync(rq1) ? rq1 : rq2;

	if ((rq1->cmd_flags ^ rq2->cmd_flags) & REQ_PRIO)
		return rq1->cmd_flags & REQ_PRIO ? rq1 : rq2;

	s1 = blk_rq_pos(rq1);
	s2 = blk_rq_pos(rq2);

	/*
	 * by definition, 1KiB is 2 sectors
	 */
	back_max = blurd->blur_back_max * 2;

	/*
	 * Strict one way elevator _except_ in the case where we allow
	 * short backward seeks which are biased as twice the cost of a
	 * similar forward seek.
	 */
	if (s1 >= last)
		d1 = s1 - last;
	else if (s1 + back_max >= last)
		d1 = (last - s1) * blurd->blur_back_penalty;
	else
		wrap |= blur_RQ1_WRAP;

	if (s2 >= last)
		d2 = s2 - last;
	else if (s2 + back_max >= last)
		d2 = (last - s2) * blurd->blur_back_penalty;
	else
		wrap |= blur_RQ2_WRAP;

	/* Found required data */

	/*
	 * By doing switch() on the bit mask "wrap" we avoid having to
	 * check two variables for all permutations: --> faster!
	 */
	switch (wrap) {
	case 0: /* common case for blur: rq1 and rq2 not wrapped */
		if (d1 < d2)
			return rq1;
		else if (d2 < d1)
			return rq2;
		else {
			if (s1 >= s2)
				return rq1;
			else
				return rq2;
		}

	case blur_RQ2_WRAP:
		return rq1;
	case blur_RQ1_WRAP:
		return rq2;
	case (blur_RQ1_WRAP|blur_RQ2_WRAP): /* both rqs wrapped */
	default:
		/*
		 * Since both rqs are wrapped,
		 * start with the one that's further behind head
		 * (--> only *one* back seek required),
		 * since back seek takes more time than forward.
		 */
		if (s1 <= s2)
			return rq1;
		else
			return rq2;
	}
}

/*
 * The below is leftmost cache rbtree addon
 */
static struct blur_queue *blur_rb_first(struct blur_rb_root *root)
{
	/* Service tree is empty */
	if (!root->count)
		return NULL;

	if (!root->left)
		root->left = rb_first(&root->rb);

	if (root->left)
		return rb_entry(root->left, struct blur_queue, rb_node);

	return NULL;
}

static struct blur_group *blur_rb_first_group(struct blur_rb_root *root)
{
	if (!root->left)
		root->left = rb_first(&root->rb);

	if (root->left)
		return rb_entry_blurg(root->left);

	return NULL;
}

static void rb_erase_init(struct rb_node *n, struct rb_root *root)
{
	rb_erase(n, root);
	RB_CLEAR_NODE(n);
}

static void blur_rb_erase(struct rb_node *n, struct blur_rb_root *root)
{
	if (root->left == n)
		root->left = NULL;
	rb_erase_init(n, &root->rb);
	--root->count;
}

/*
 * would be nice to take fifo expire time into account as well
 */
static struct request *
blur_find_next_rq(struct blur_data *blurd, struct blur_queue *blurq,
		  struct request *last)
{
	struct rb_node *rbnext = rb_next(&last->rb_node);
	struct rb_node *rbprev = rb_prev(&last->rb_node);
	struct request *next = NULL, *prev = NULL;

	BUG_ON(RB_EMPTY_NODE(&last->rb_node));

	if (rbprev)
		prev = rb_entry_rq(rbprev);

	if (rbnext)
		next = rb_entry_rq(rbnext);
	else {
		rbnext = rb_first(&blurq->sort_list);
		if (rbnext && rbnext != &last->rb_node)
			next = rb_entry_rq(rbnext);
	}

	return blur_choose_req(blurd, next, prev, blk_rq_pos(last));
}

static unsigned long blur_slice_offset(struct blur_data *blurd,
				      struct blur_queue *blurq)
{
	/*
	 * just an approximation, should be ok.
	 */
	return (blurq->blurg->nr_blurq - 1) * (blur_prio_slice(blurd, 1, 0) -
		       blur_prio_slice(blurd, blur_blurq_sync(blurq), blurq->ioprio));
}

static inline s64
blurg_key(struct blur_rb_root *st, struct blur_group *blurg)
{
	return blurg->vdisktime - st->min_vdisktime;
}

static void
__blur_group_service_tree_add(struct blur_rb_root *st, struct blur_group *blurg)
{
	struct rb_node **node = &st->rb.rb_node;
	struct rb_node *parent = NULL;
	struct blur_group *__blurg;
	s64 key = blurg_key(st, blurg);
	int left = 1;

	while (*node != NULL) {
		parent = *node;
		__blurg = rb_entry_blurg(parent);

		if (key < blurg_key(st, __blurg))
			node = &parent->rb_left;
		else {
			node = &parent->rb_right;
			left = 0;
		}
	}

	if (left)
		st->left = &blurg->rb_node;

	rb_link_node(&blurg->rb_node, parent, node);
	rb_insert_color(&blurg->rb_node, &st->rb);
}

static void
blur_update_group_weight(struct blur_group *blurg)
{
	BUG_ON(!RB_EMPTY_NODE(&blurg->rb_node));
	if (blurg->needs_update) {
		blurg->weight = blurg->new_weight;
		blurg->needs_update = false;
	}
}

static void
blur_group_service_tree_add(struct blur_rb_root *st, struct blur_group *blurg)
{
	BUG_ON(!RB_EMPTY_NODE(&blurg->rb_node));

	blur_update_group_weight(blurg);
	__blur_group_service_tree_add(st, blurg);
	st->total_weight += blurg->weight;
}

static void
blur_group_notify_queue_add(struct blur_data *blurd, struct blur_group *blurg)
{
	struct blur_rb_root *st = &blurd->grp_service_tree;
	struct blur_group *__blurg;
	struct rb_node *n;

	blurg->nr_blurq++;
	if (!RB_EMPTY_NODE(&blurg->rb_node))
		return;

	/*
	 * Currently put the group at the end. Later implement something
	 * so that groups get lesser vtime based on their weights, so that
	 * if group does not loose all if it was not continuously backlogged.
	 */
	n = rb_last(&st->rb);
	if (n) {
		__blurg = rb_entry_blurg(n);
		blurg->vdisktime = __blurg->vdisktime + blur_IDLE_DELAY;
	} else
		blurg->vdisktime = st->min_vdisktime;
	blur_group_service_tree_add(st, blurg);
}

static void
blur_group_service_tree_del(struct blur_rb_root *st, struct blur_group *blurg)
{
	st->total_weight -= blurg->weight;
	if (!RB_EMPTY_NODE(&blurg->rb_node))
		blur_rb_erase(&blurg->rb_node, st);
}

static void
blur_group_notify_queue_del(struct blur_data *blurd, struct blur_group *blurg)
{
	struct blur_rb_root *st = &blurd->grp_service_tree;

	BUG_ON(blurg->nr_blurq < 1);
	blurg->nr_blurq--;

	/* If there are other blur queues under this group, don't delete it */
	if (blurg->nr_blurq)
		return;

	blur_log_blurg(blurd, blurg, "del_from_rr group");
	blur_group_service_tree_del(st, blurg);
	blurg->saved_workload_slice = 0;
	blur_blkiocg_update_dequeue_stats(&blurg->blkg, 1);
}

static inline unsigned int blur_blurq_slice_usage(struct blur_queue *blurq,
						unsigned int *unaccounted_time)
{
	unsigned int slice_used;

	/*
	 * Queue got expired before even a single request completed or
	 * got expired immediately after first request completion.
	 */
	if (!blurq->slice_start || blurq->slice_start == jiffies) {
		/*
		 * Also charge the seek time incurred to the group, otherwise
		 * if there are mutiple queues in the group, each can dispatch
		 * a single request on seeky media and cause lots of seek time
		 * and group will never know it.
		 */
		slice_used = max_t(unsigned, (jiffies - blurq->dispatch_start),
					1);
	} else {
		slice_used = jiffies - blurq->slice_start;
		if (slice_used > blurq->allocated_slice) {
			*unaccounted_time = slice_used - blurq->allocated_slice;
			slice_used = blurq->allocated_slice;
		}
		if (time_after(blurq->slice_start, blurq->dispatch_start))
			*unaccounted_time += blurq->slice_start -
					blurq->dispatch_start;
	}

	return slice_used;
}

static void blur_group_served(struct blur_data *blurd, struct blur_group *blurg,
				struct blur_queue *blurq)
{
	struct blur_rb_root *st = &blurd->grp_service_tree;
	unsigned int used_sl, charge, unaccounted_sl = 0;
	int nr_sync = blurg->nr_blurq - blurg_busy_async_queues(blurd, blurg)
			- blurg->service_tree_idle.count;

	BUG_ON(nr_sync < 0);
	used_sl = charge = blur_blurq_slice_usage(blurq, &unaccounted_sl);

	if (iops_mode(blurd))
		charge = blurq->slice_dispatch;
	else if (!blur_blurq_sync(blurq) && !nr_sync)
		charge = blurq->allocated_slice;

	/* Can't update vdisktime while group is on service tree */
	blur_group_service_tree_del(st, blurg);
	blurg->vdisktime += blur_scale_slice(charge, blurg);
	/* If a new weight was requested, update now, off tree */
	blur_group_service_tree_add(st, blurg);

	/* This group is being expired. Save the context */
	if (time_after(blurd->workload_expires, jiffies)) {
		blurg->saved_workload_slice = blurd->workload_expires
						- jiffies;
		blurg->saved_workload = blurd->serving_type;
		blurg->saved_serving_prio = blurd->serving_prio;
	} else
		blurg->saved_workload_slice = 0;

	blur_log_blurg(blurd, blurg, "served: vt=%llu min_vt=%llu", blurg->vdisktime,
					st->min_vdisktime);
	blur_log_blurq(blurq->blurd, blurq,
		     "sl_used=%u disp=%u charge=%u iops=%u sect=%lu",
		     used_sl, blurq->slice_dispatch, charge,
		     iops_mode(blurd), blurq->nr_sectors);
	blur_blkiocg_update_timeslice_used(&blurg->blkg, used_sl,
					  unaccounted_sl);
	blur_blkiocg_set_start_empty_time(&blurg->blkg);
}

#ifdef CONFIG_blur_GROUP_IOSCHED
static inline struct blur_group *blurg_of_blkg(struct blkio_group *blkg)
{
	if (blkg)
		return container_of(blkg, struct blur_group, blkg);
	return NULL;
}

static void blur_update_blkio_group_weight(void *key, struct blkio_group *blkg,
					  unsigned int weight)
{
	struct blur_group *blurg = blurg_of_blkg(blkg);
	blurg->new_weight = weight;
	blurg->needs_update = true;
}

static void blur_init_add_blurg_lists(struct blur_data *blurd,
			struct blur_group *blurg, struct blkio_cgroup *blkcg)
{
	struct backing_dev_info *bdi = &blurd->queue->backing_dev_info;
	unsigned int major, minor;

	/*
	 * Add group onto cgroup list. It might happen that bdi->dev is
	 * not initialized yet. Initialize this new group without major
	 * and minor info and this info will be filled in once a new thread
	 * comes for IO.
	 */
	if (bdi->dev) {
		sscanf(dev_name(bdi->dev), "%u:%u", &major, &minor);
		blur_blkiocg_add_blkio_group(blkcg, &blurg->blkg,
					(void *)blurd, MKDEV(major, minor));
	} else
		blur_blkiocg_add_blkio_group(blkcg, &blurg->blkg,
					(void *)blurd, 0);

	blurd->nr_blkcg_linked_grps++;
	blurg->weight = blkcg_get_weight(blkcg, blurg->blkg.dev);

	/* Add group on blurd list */
	hlist_add_head(&blurg->blurd_node, &blurd->blurg_list);
}

/*
 * Should be called from sleepable context. No request queue lock as per
 * cpu stats are allocated dynamically and alloc_percpu needs to be called
 * from sleepable context.
 */
static struct blur_group * blur_alloc_blurg(struct blur_data *blurd)
{
	struct blur_group *blurg = NULL;
	int i, j, ret;
	struct blur_rb_root *st;

	blurg = kzalloc_node(sizeof(*blurg), GFP_ATOMIC, blurd->queue->node);
	if (!blurg)
		return NULL;

	for_each_blurg_st(blurg, i, j, st)
		*st = blur_RB_ROOT;
	RB_CLEAR_NODE(&blurg->rb_node);

	blurg->ttime.last_end_request = jiffies;

	/*
	 * Take the initial reference that will be released on destroy
	 * This can be thought of a joint reference by cgroup and
	 * elevator which will be dropped by either elevator exit
	 * or cgroup deletion path depending on who is exiting first.
	 */
	blurg->ref = 1;

	ret = blkio_alloc_blkg_stats(&blurg->blkg);
	if (ret) {
		kfree(blurg);
		return NULL;
	}

	return blurg;
}

static struct blur_group *
blur_find_blurg(struct blur_data *blurd, struct blkio_cgroup *blkcg)
{
	struct blur_group *blurg = NULL;
	void *key = blurd;
	struct backing_dev_info *bdi = &blurd->queue->backing_dev_info;
	unsigned int major, minor;

	/*
	 * This is the common case when there are no blkio cgroups.
	 * Avoid lookup in this case
	 */
	if (blkcg == &blkio_root_cgroup)
		blurg = &blurd->root_group;
	else
		blurg = blurg_of_blkg(blkiocg_lookup_group(blkcg, key));

	if (blurg && !blurg->blkg.dev && bdi->dev && dev_name(bdi->dev)) {
		sscanf(dev_name(bdi->dev), "%u:%u", &major, &minor);
		blurg->blkg.dev = MKDEV(major, minor);
	}

	return blurg;
}

/*
 * Search for the blur group current task belongs to. request_queue lock must
 * be held.
 */
static struct blur_group *blur_get_blurg(struct blur_data *blurd)
{
	struct blkio_cgroup *blkcg;
	struct blur_group *blurg = NULL, *__blurg = NULL;
	struct request_queue *q = blurd->queue;

	rcu_read_lock();
	blkcg = task_blkio_cgroup(current);
	blurg = blur_find_blurg(blurd, blkcg);
	if (blurg) {
		rcu_read_unlock();
		return blurg;
	}

	/*
	 * Need to allocate a group. Allocation of group also needs allocation
	 * of per cpu stats which in-turn takes a mutex() and can block. Hence
	 * we need to drop rcu lock and queue_lock before we call alloc.
	 *
	 * Not taking any queue reference here and assuming that queue is
	 * around by the time we return. blur queue allocation code does
	 * the same. It might be racy though.
	 */

	rcu_read_unlock();
	spin_unlock_irq(q->queue_lock);

	blurg = blur_alloc_blurg(blurd);

	spin_lock_irq(q->queue_lock);

	rcu_read_lock();
	blkcg = task_blkio_cgroup(current);

	/*
	 * If some other thread already allocated the group while we were
	 * not holding queue lock, free up the group
	 */
	__blurg = blur_find_blurg(blurd, blkcg);

	if (__blurg) {
		kfree(blurg);
		rcu_read_unlock();
		return __blurg;
	}

	if (!blurg)
		blurg = &blurd->root_group;

	blur_init_add_blurg_lists(blurd, blurg, blkcg);
	rcu_read_unlock();
	return blurg;
}

static inline struct blur_group *blur_ref_get_blurg(struct blur_group *blurg)
{
	blurg->ref++;
	return blurg;
}

static void blur_link_blurq_blurg(struct blur_queue *blurq, struct blur_group *blurg)
{
	/* Currently, all async queues are mapped to root group */
	if (!blur_blurq_sync(blurq))
		blurg = &blurq->blurd->root_group;

	blurq->blurg = blurg;
	/* blurq reference on blurg */
	blurq->blurg->ref++;
}

static void blur_put_blurg(struct blur_group *blurg)
{
	struct blur_rb_root *st;
	int i, j;

	BUG_ON(blurg->ref <= 0);
	blurg->ref--;
	if (blurg->ref)
		return;
	for_each_blurg_st(blurg, i, j, st)
		BUG_ON(!RB_EMPTY_ROOT(&st->rb));
	free_percpu(blurg->blkg.stats_cpu);
	kfree(blurg);
}

static void blur_destroy_blurg(struct blur_data *blurd, struct blur_group *blurg)
{
	/* Something wrong if we are trying to remove same group twice */
	BUG_ON(hlist_unhashed(&blurg->blurd_node));

	hlist_del_init(&blurg->blurd_node);

	BUG_ON(blurd->nr_blkcg_linked_grps <= 0);
	blurd->nr_blkcg_linked_grps--;

	/*
	 * Put the reference taken at the time of creation so that when all
	 * queues are gone, group can be destroyed.
	 */
	blur_put_blurg(blurg);
}

static void blur_release_blur_groups(struct blur_data *blurd)
{
	struct hlist_node *pos, *n;
	struct blur_group *blurg;

	hlist_for_each_entry_safe(blurg, pos, n, &blurd->blurg_list, blurd_node) {
		/*
		 * If cgroup removal path got to blk_group first and removed
		 * it from cgroup list, then it will take care of destroying
		 * blurg also.
		 */
		if (!blur_blkiocg_del_blkio_group(&blurg->blkg))
			blur_destroy_blurg(blurd, blurg);
	}
}

/*
 * Blk cgroup controller notification saying that blkio_group object is being
 * delinked as associated cgroup object is going away. That also means that
 * no new IO will come in this group. So get rid of this group as soon as
 * any pending IO in the group is finished.
 *
 * This function is called under rcu_read_lock(). key is the rcu protected
 * pointer. That means "key" is a valid blur_data pointer as long as we are rcu
 * read lock.
 *
 * "key" was fetched from blkio_group under blkio_cgroup->lock. That means
 * it should not be NULL as even if elevator was exiting, cgroup deltion
 * path got to it first.
 */
static void blur_unlink_blkio_group(void *key, struct blkio_group *blkg)
{
	unsigned long  flags;
	struct blur_data *blurd = key;

	spin_lock_irqsave(blurd->queue->queue_lock, flags);
	blur_destroy_blurg(blurd, blurg_of_blkg(blkg));
	spin_unlock_irqrestore(blurd->queue->queue_lock, flags);
}

#else /* GROUP_IOSCHED */
static struct blur_group *blur_get_blurg(struct blur_data *blurd)
{
	return &blurd->root_group;
}

static inline struct blur_group *blur_ref_get_blurg(struct blur_group *blurg)
{
	return blurg;
}

static inline void
blur_link_blurq_blurg(struct blur_queue *blurq, struct blur_group *blurg) {
	blurq->blurg = blurg;
}

static void blur_release_blur_groups(struct blur_data *blurd) {}
static inline void blur_put_blurg(struct blur_group *blurg) {}

#endif /* GROUP_IOSCHED */

/*
 * The blurd->service_trees holds all pending blur_queue's that have
 * requests waiting to be processed. It is sorted in the order that
 * we will service the queues.
 */
static void blur_service_tree_add(struct blur_data *blurd, struct blur_queue *blurq,
				 bool add_front)
{
	struct rb_node **p, *parent;
	struct blur_queue *__blurq;
	unsigned long rb_key;
	struct blur_rb_root *service_tree;
	int left;
	int new_blurq = 1;

	service_tree = service_tree_for(blurq->blurg, blurq_prio(blurq),
						blurq_type(blurq));
	if (blur_class_idle(blurq)) {
		rb_key = blur_IDLE_DELAY;
		parent = rb_last(&service_tree->rb);
		if (parent && parent != &blurq->rb_node) {
			__blurq = rb_entry(parent, struct blur_queue, rb_node);
			rb_key += __blurq->rb_key;
		} else
			rb_key += jiffies;
	} else if (!add_front) {
		/*
		 * Get our rb key offset. Subtract any residual slice
		 * value carried from last service. A negative resid
		 * count indicates slice overrun, and this should position
		 * the next service time further away in the tree.
		 */
		rb_key = blur_slice_offset(blurd, blurq) + jiffies;
		rb_key -= blurq->slice_resid;
		blurq->slice_resid = 0;
	} else {
		rb_key = -HZ;
		__blurq = blur_rb_first(service_tree);
		rb_key += __blurq ? __blurq->rb_key : jiffies;
	}

	if (!RB_EMPTY_NODE(&blurq->rb_node)) {
		new_blurq = 0;
		/*
		 * same position, nothing more to do
		 */
		if (rb_key == blurq->rb_key &&
		    blurq->service_tree == service_tree)
			return;

		blur_rb_erase(&blurq->rb_node, blurq->service_tree);
		blurq->service_tree = NULL;
	}

	left = 1;
	parent = NULL;
	blurq->service_tree = service_tree;
	p = &service_tree->rb.rb_node;
	while (*p) {
		struct rb_node **n;

		parent = *p;
		__blurq = rb_entry(parent, struct blur_queue, rb_node);

		/*
		 * sort by key, that represents service time.
		 */
		if (time_before(rb_key, __blurq->rb_key))
			n = &(*p)->rb_left;
		else {
			n = &(*p)->rb_right;
			left = 0;
		}

		p = n;
	}

	if (left)
		service_tree->left = &blurq->rb_node;

	blurq->rb_key = rb_key;
	rb_link_node(&blurq->rb_node, parent, p);
	rb_insert_color(&blurq->rb_node, &service_tree->rb);
	service_tree->count++;
	if (add_front || !new_blurq)
		return;
	blur_group_notify_queue_add(blurd, blurq->blurg);
}

static struct blur_queue *
blur_prio_tree_lookup(struct blur_data *blurd, struct rb_root *root,
		     sector_t sector, struct rb_node **ret_parent,
		     struct rb_node ***rb_link)
{
	struct rb_node **p, *parent;
	struct blur_queue *blurq = NULL;

	parent = NULL;
	p = &root->rb_node;
	while (*p) {
		struct rb_node **n;

		parent = *p;
		blurq = rb_entry(parent, struct blur_queue, p_node);

		/*
		 * Sort strictly based on sector.  Smallest to the left,
		 * largest to the right.
		 */
		if (sector > blk_rq_pos(blurq->next_rq))
			n = &(*p)->rb_right;
		else if (sector < blk_rq_pos(blurq->next_rq))
			n = &(*p)->rb_left;
		else
			break;
		p = n;
		blurq = NULL;
	}

	*ret_parent = parent;
	if (rb_link)
		*rb_link = p;
	return blurq;
}

static void blur_prio_tree_add(struct blur_data *blurd, struct blur_queue *blurq)
{
	struct rb_node **p, *parent;
	struct blur_queue *__blurq;

	if (blurq->p_root) {
		rb_erase(&blurq->p_node, blurq->p_root);
		blurq->p_root = NULL;
	}

	if (blur_class_idle(blurq))
		return;
	if (!blurq->next_rq)
		return;

	blurq->p_root = &blurd->prio_trees[blurq->org_ioprio];
	__blurq = blur_prio_tree_lookup(blurd, blurq->p_root,
				      blk_rq_pos(blurq->next_rq), &parent, &p);
	if (!__blurq) {
		rb_link_node(&blurq->p_node, parent, p);
		rb_insert_color(&blurq->p_node, blurq->p_root);
	} else
		blurq->p_root = NULL;
}

/*
 * Update blurq's position in the service tree.
 */
static void blur_resort_rr_list(struct blur_data *blurd, struct blur_queue *blurq)
{
	/*
	 * Resorting requires the blurq to be on the RR list already.
	 */
	if (blur_blurq_on_rr(blurq)) {
		blur_service_tree_add(blurd, blurq, 0);
		blur_prio_tree_add(blurd, blurq);
	}
}

/*
 * add to busy list of queues for service, trying to be fair in ordering
 * the pending list according to last request service
 */
static void blur_add_blurq_rr(struct blur_data *blurd, struct blur_queue *blurq)
{
	blur_log_blurq(blurd, blurq, "add_to_rr");
	BUG_ON(blur_blurq_on_rr(blurq));
	blur_mark_blurq_on_rr(blurq);
	blurd->busy_queues++;
	if (blur_blurq_sync(blurq))
		blurd->busy_sync_queues++;

	blur_resort_rr_list(blurd, blurq);
}

/*
 * Called when the blurq no longer has requests pending, remove it from
 * the service tree.
 */
static void blur_del_blurq_rr(struct blur_data *blurd, struct blur_queue *blurq)
{
	blur_log_blurq(blurd, blurq, "del_from_rr");
	BUG_ON(!blur_blurq_on_rr(blurq));
	blur_clear_blurq_on_rr(blurq);

	if (!RB_EMPTY_NODE(&blurq->rb_node)) {
		blur_rb_erase(&blurq->rb_node, blurq->service_tree);
		blurq->service_tree = NULL;
	}
	if (blurq->p_root) {
		rb_erase(&blurq->p_node, blurq->p_root);
		blurq->p_root = NULL;
	}

	blur_group_notify_queue_del(blurd, blurq->blurg);
	BUG_ON(!blurd->busy_queues);
	blurd->busy_queues--;
	if (blur_blurq_sync(blurq))
		blurd->busy_sync_queues--;
}

/*
 * rb tree support functions
 */
static void blur_del_rq_rb(struct request *rq)
{
	struct blur_queue *blurq = RQ_blurQ(rq);
	const int sync = rq_is_sync(rq);

	BUG_ON(!blurq->queued[sync]);
	blurq->queued[sync]--;

	elv_rb_del(&blurq->sort_list, rq);

	if (blur_blurq_on_rr(blurq) && RB_EMPTY_ROOT(&blurq->sort_list)) {
		/*
		 * Queue will be deleted from service tree when we actually
		 * expire it later. Right now just remove it from prio tree
		 * as it is empty.
		 */
		if (blurq->p_root) {
			rb_erase(&blurq->p_node, blurq->p_root);
			blurq->p_root = NULL;
		}
	}
}

static void blur_add_rq_rb(struct request *rq)
{
	struct blur_queue *blurq = RQ_blurQ(rq);
	struct blur_data *blurd = blurq->blurd;
	struct request *prev;

	blurq->queued[rq_is_sync(rq)]++;

	elv_rb_add(&blurq->sort_list, rq);

	if (!blur_blurq_on_rr(blurq))
		blur_add_blurq_rr(blurd, blurq);

	/*
	 * check if this request is a better next-serve candidate
	 */
	prev = blurq->next_rq;
	blurq->next_rq = blur_choose_req(blurd, blurq->next_rq, rq, blurd->last_position);

	/*
	 * adjust priority tree position, if ->next_rq changes
	 */
	if (prev != blurq->next_rq)
		blur_prio_tree_add(blurd, blurq);

	BUG_ON(!blurq->next_rq);
}

static void blur_reposition_rq_rb(struct blur_queue *blurq, struct request *rq)
{
	elv_rb_del(&blurq->sort_list, rq);
	blurq->queued[rq_is_sync(rq)]--;
	blur_blkiocg_update_io_remove_stats(&(RQ_blurG(rq))->blkg,
					rq_data_dir(rq), rq_is_sync(rq));
	blur_add_rq_rb(rq);
	blur_blkiocg_update_io_add_stats(&(RQ_blurG(rq))->blkg,
			&blurq->blurd->serving_group->blkg, rq_data_dir(rq),
			rq_is_sync(rq));
}

static struct request *
blur_find_rq_fmerge(struct blur_data *blurd, struct bio *bio)
{
	struct task_struct *tsk = current;
	struct blur_io_cq *cic;
	struct blur_queue *blurq;

	cic = blur_cic_lookup(blurd, tsk->io_context);
	if (!cic)
		return NULL;

	blurq = cic_to_blurq(cic, blur_bio_sync(bio));
	if (blurq) {
		sector_t sector = bio->bi_sector + bio_sectors(bio);

		return elv_rb_find(&blurq->sort_list, sector);
	}

	return NULL;
}

static void blur_activate_request(struct request_queue *q, struct request *rq)
{
	struct blur_data *blurd = q->elevator->elevator_data;

	blurd->rq_in_driver++;
	blur_log_blurq(blurd, RQ_blurQ(rq), "activate rq, drv=%d",
						blurd->rq_in_driver);

	blurd->last_position = blk_rq_pos(rq) + blk_rq_sectors(rq);
}

static void blur_deactivate_request(struct request_queue *q, struct request *rq)
{
	struct blur_data *blurd = q->elevator->elevator_data;

	WARN_ON(!blurd->rq_in_driver);
	blurd->rq_in_driver--;
	blur_log_blurq(blurd, RQ_blurQ(rq), "deactivate rq, drv=%d",
						blurd->rq_in_driver);
}

static void blur_remove_request(struct request *rq)
{
	struct blur_queue *blurq = RQ_blurQ(rq);

	if (blurq->next_rq == rq)
		blurq->next_rq = blur_find_next_rq(blurq->blurd, blurq, rq);

	list_del_init(&rq->queuelist);
	blur_del_rq_rb(rq);

	blurq->blurd->rq_queued--;
	blur_blkiocg_update_io_remove_stats(&(RQ_blurG(rq))->blkg,
					rq_data_dir(rq), rq_is_sync(rq));
	if (rq->cmd_flags & REQ_PRIO) {
		WARN_ON(!blurq->prio_pending);
		blurq->prio_pending--;
	}
}

static int blur_merge(struct request_queue *q, struct request **req,
		     struct bio *bio)
{
	struct blur_data *blurd = q->elevator->elevator_data;
	struct request *__rq;

	__rq = blur_find_rq_fmerge(blurd, bio);
	if (__rq && elv_rq_merge_ok(__rq, bio)) {
		*req = __rq;
		return ELEVATOR_FRONT_MERGE;
	}

	return ELEVATOR_NO_MERGE;
}

static void blur_merged_request(struct request_queue *q, struct request *req,
			       int type)
{
	if (type == ELEVATOR_FRONT_MERGE) {
		struct blur_queue *blurq = RQ_blurQ(req);

		blur_reposition_rq_rb(blurq, req);
	}
}

static void blur_bio_merged(struct request_queue *q, struct request *req,
				struct bio *bio)
{
	blur_blkiocg_update_io_merged_stats(&(RQ_blurG(req))->blkg,
					bio_data_dir(bio), blur_bio_sync(bio));
}

static void
blur_merged_requests(struct request_queue *q, struct request *rq,
		    struct request *next)
{
	struct blur_queue *blurq = RQ_blurQ(rq);
	struct blur_data *blurd = q->elevator->elevator_data;

	/*
	 * reposition in fifo if next is older than rq
	 */
	if (!list_empty(&rq->queuelist) && !list_empty(&next->queuelist) &&
	    time_before(rq_fifo_time(next), rq_fifo_time(rq))) {
		list_move(&rq->queuelist, &next->queuelist);
		rq_set_fifo_time(rq, rq_fifo_time(next));
	}

	if (blurq->next_rq == next)
		blurq->next_rq = rq;
	blur_remove_request(next);
	blur_blkiocg_update_io_merged_stats(&(RQ_blurG(rq))->blkg,
					rq_data_dir(next), rq_is_sync(next));

	blurq = RQ_blurQ(next);
	/*
	 * all requests of this queue are merged to other queues, delete it
	 * from the service tree. If it's the active_queue,
	 * blur_dispatch_requests() will choose to expire it or do idle
	 */
	if (blur_blurq_on_rr(blurq) && RB_EMPTY_ROOT(&blurq->sort_list) &&
	    blurq != blurd->active_queue)
		blur_del_blurq_rr(blurd, blurq);
}

static int blur_allow_merge(struct request_queue *q, struct request *rq,
			   struct bio *bio)
{
	struct blur_data *blurd = q->elevator->elevator_data;
	struct blur_io_cq *cic;
	struct blur_queue *blurq;

	/*
	 * Disallow merge of a sync bio into an async request.
	 */
	if (blur_bio_sync(bio) && !rq_is_sync(rq))
		return false;

	/*
	 * Lookup the blurq that this bio will be queued with and allow
	 * merge only if rq is queued there.
	 */
	cic = blur_cic_lookup(blurd, current->io_context);
	if (!cic)
		return false;

	blurq = cic_to_blurq(cic, blur_bio_sync(bio));
	return blurq == RQ_blurQ(rq);
}

static inline void blur_del_timer(struct blur_data *blurd, struct blur_queue *blurq)
{
	del_timer(&blurd->idle_slice_timer);
	blur_blkiocg_update_idle_time_stats(&blurq->blurg->blkg);
}

static void __blur_set_active_queue(struct blur_data *blurd,
				   struct blur_queue *blurq)
{
	if (blurq) {
		blur_log_blurq(blurd, blurq, "set_active wl_prio:%d wl_type:%d",
				blurd->serving_prio, blurd->serving_type);
		blur_blkiocg_update_avg_queue_size_stats(&blurq->blurg->blkg);
		blurq->slice_start = 0;
		blurq->dispatch_start = jiffies;
		blurq->allocated_slice = 0;
		blurq->slice_end = 0;
		blurq->slice_dispatch = 0;
		blurq->nr_sectors = 0;

		blur_clear_blurq_wait_request(blurq);
		blur_clear_blurq_must_dispatch(blurq);
		blur_clear_blurq_must_alloc_slice(blurq);
		blur_clear_blurq_fifo_expire(blurq);
		blur_mark_blurq_slice_new(blurq);

		blur_del_timer(blurd, blurq);
	}

	blurd->active_queue = blurq;
}

/*
 * current blurq expired its slice (or was too idle), select new one
 */
static void
__blur_slice_expired(struct blur_data *blurd, struct blur_queue *blurq,
		    bool timed_out)
{
	blur_log_blurq(blurd, blurq, "slice expired t=%d", timed_out);

	if (blur_blurq_wait_request(blurq))
		blur_del_timer(blurd, blurq);

	blur_clear_blurq_wait_request(blurq);
	blur_clear_blurq_wait_busy(blurq);

	/*
	 * If this blurq is shared between multiple processes, check to
	 * make sure that those processes are still issuing I/Os within
	 * the mean seek distance.  If not, it may be time to break the
	 * queues apart again.
	 */
	if (blur_blurq_coop(blurq) && blurQ_SEEKY(blurq))
		blur_mark_blurq_split_coop(blurq);

	/*
	 * store what was left of this slice, if the queue idled/timed out
	 */
	if (timed_out) {
		if (blur_blurq_slice_new(blurq))
			blurq->slice_resid = blur_scaled_blurq_slice(blurd, blurq);
		else
			blurq->slice_resid = blurq->slice_end - jiffies;
		blur_log_blurq(blurd, blurq, "resid=%ld", blurq->slice_resid);
	}

	blur_group_served(blurd, blurq->blurg, blurq);

	if (blur_blurq_on_rr(blurq) && RB_EMPTY_ROOT(&blurq->sort_list))
		blur_del_blurq_rr(blurd, blurq);

	blur_resort_rr_list(blurd, blurq);

	if (blurq == blurd->active_queue)
		blurd->active_queue = NULL;

	if (blurd->active_cic) {
		put_io_context(blurd->active_cic->icq.ioc);
		blurd->active_cic = NULL;
	}
}

static inline void blur_slice_expired(struct blur_data *blurd, bool timed_out)
{
	struct blur_queue *blurq = blurd->active_queue;

	if (blurq)
		__blur_slice_expired(blurd, blurq, timed_out);
}

/*
 * Get next queue for service. Unless we have a queue preemption,
 * we'll simply select the first blurq in the service tree.
 */
static struct blur_queue *blur_get_next_queue(struct blur_data *blurd)
{
	struct blur_rb_root *service_tree =
		service_tree_for(blurd->serving_group, blurd->serving_prio,
					blurd->serving_type);

	if (!blurd->rq_queued)
		return NULL;

	/* There is nothing to dispatch */
	if (!service_tree)
		return NULL;
	if (RB_EMPTY_ROOT(&service_tree->rb))
		return NULL;
	return blur_rb_first(service_tree);
}

static struct blur_queue *blur_get_next_queue_forced(struct blur_data *blurd)
{
	struct blur_group *blurg;
	struct blur_queue *blurq;
	int i, j;
	struct blur_rb_root *st;

	if (!blurd->rq_queued)
		return NULL;

	blurg = blur_get_next_blurg(blurd);
	if (!blurg)
		return NULL;

	for_each_blurg_st(blurg, i, j, st)
		if ((blurq = blur_rb_first(st)) != NULL)
			return blurq;
	return NULL;
}

/*
 * Get and set a new active queue for service.
 */
static struct blur_queue *blur_set_active_queue(struct blur_data *blurd,
					      struct blur_queue *blurq)
{
	if (!blurq)
		blurq = blur_get_next_queue(blurd);

	__blur_set_active_queue(blurd, blurq);
	return blurq;
}

static inline sector_t blur_dist_from_last(struct blur_data *blurd,
					  struct request *rq)
{
	if (blk_rq_pos(rq) >= blurd->last_position)
		return blk_rq_pos(rq) - blurd->last_position;
	else
		return blurd->last_position - blk_rq_pos(rq);
}

static inline int blur_rq_close(struct blur_data *blurd, struct blur_queue *blurq,
			       struct request *rq)
{
	return blur_dist_from_last(blurd, rq) <= blurQ_CLOSE_THR;
}

static struct blur_queue *blurq_close(struct blur_data *blurd,
				    struct blur_queue *cur_blurq)
{
	struct rb_root *root = &blurd->prio_trees[cur_blurq->org_ioprio];
	struct rb_node *parent, *node;
	struct blur_queue *__blurq;
	sector_t sector = blurd->last_position;

	if (RB_EMPTY_ROOT(root))
		return NULL;

	/*
	 * First, if we find a request starting at the end of the last
	 * request, choose it.
	 */
	__blurq = blur_prio_tree_lookup(blurd, root, sector, &parent, NULL);
	if (__blurq)
		return __blurq;

	/*
	 * If the exact sector wasn't found, the parent of the NULL leaf
	 * will contain the closest sector.
	 */
	__blurq = rb_entry(parent, struct blur_queue, p_node);
	if (blur_rq_close(blurd, cur_blurq, __blurq->next_rq))
		return __blurq;

	if (blk_rq_pos(__blurq->next_rq) < sector)
		node = rb_next(&__blurq->p_node);
	else
		node = rb_prev(&__blurq->p_node);
	if (!node)
		return NULL;

	__blurq = rb_entry(node, struct blur_queue, p_node);
	if (blur_rq_close(blurd, cur_blurq, __blurq->next_rq))
		return __blurq;

	return NULL;
}

/*
 * blurd - obvious
 * cur_blurq - passed in so that we don't decide that the current queue is
 * 	      closely cooperating with itself.
 *
 * So, basically we're assuming that that cur_blurq has dispatched at least
 * one request, and that blurd->last_position reflects a position on the disk
 * associated with the I/O issued by cur_blurq.  I'm not sure this is a valid
 * assumption.
 */
static struct blur_queue *blur_close_cooperator(struct blur_data *blurd,
					      struct blur_queue *cur_blurq)
{
	struct blur_queue *blurq;

	if (blur_class_idle(cur_blurq))
		return NULL;
	if (!blur_blurq_sync(cur_blurq))
		return NULL;
	if (blurQ_SEEKY(cur_blurq))
		return NULL;

	/*
	 * Don't search priority tree if it's the only queue in the group.
	 */
	if (cur_blurq->blurg->nr_blurq == 1)
		return NULL;

	/*
	 * We should notice if some of the queues are cooperating, eg
	 * working closely on the same area of the disk. In that case,
	 * we can group them together and don't waste time idling.
	 */
	blurq = blurq_close(blurd, cur_blurq);
	if (!blurq)
		return NULL;

	/* If new queue belongs to different blur_group, don't choose it */
	if (cur_blurq->blurg != blurq->blurg)
		return NULL;

	/*
	 * It only makes sense to merge sync queues.
	 */
	if (!blur_blurq_sync(blurq))
		return NULL;
	if (blurQ_SEEKY(blurq))
		return NULL;

	/*
	 * Do not merge queues of different priority classes
	 */
	if (blur_class_rt(blurq) != blur_class_rt(cur_blurq))
		return NULL;

	return blurq;
}

/*
 * Determine whether we should enforce idle window for this queue.
 */

static bool blur_should_idle(struct blur_data *blurd, struct blur_queue *blurq)
{
	enum wl_prio_t prio = blurq_prio(blurq);
	struct blur_rb_root *service_tree = blurq->service_tree;

	BUG_ON(!service_tree);
	BUG_ON(!service_tree->count);

	if (!blurd->blur_slice_idle)
		return false;

	/* We never do for idle class queues. */
	if (prio == IDLE_WORKLOAD)
		return false;

	/* We do for queues that were marked with idle window flag. */
	if (blur_blurq_idle_window(blurq) &&
	   !(blk_queue_nonrot(blurd->queue) && blurd->hw_tag))
		return true;

	/*
	 * Otherwise, we do only if they are the last ones
	 * in their service tree.
	 */
	if (service_tree->count == 1 && blur_blurq_sync(blurq) &&
	   !blur_io_thinktime_big(blurd, &service_tree->ttime, false))
		return true;
	blur_log_blurq(blurd, blurq, "Not idling. st->count:%d",
			service_tree->count);
	return false;
}

static void blur_arm_slice_timer(struct blur_data *blurd)
{
	struct blur_queue *blurq = blurd->active_queue;
	struct blur_io_cq *cic;
	unsigned long sl, group_idle = 0;

	/*
	 * SSD device without seek penalty, disable idling. But only do so
	 * for devices that support queuing, otherwise we still have a problem
	 * with sync vs async workloads.
	 */
	if (blk_queue_nonrot(blurd->queue) && blurd->hw_tag)
		return;

	WARN_ON(!RB_EMPTY_ROOT(&blurq->sort_list));
	WARN_ON(blur_blurq_slice_new(blurq));

	/*
	 * idle is disabled, either manually or by past process history
	 */
	if (!blur_should_idle(blurd, blurq)) {
		/* no queue idling. Check for group idling */
		if (blurd->blur_group_idle)
			group_idle = blurd->blur_group_idle;
		else
			return;
	}

	/*
	 * still active requests from this queue, don't idle
	 */
	if (blurq->dispatched)
		return;

	/*
	 * task has exited, don't wait
	 */
	cic = blurd->active_cic;
	if (!cic || !atomic_read(&cic->icq.ioc->nr_tasks))
		return;

	/*
	 * If our average think time is larger than the remaining time
	 * slice, then don't idle. This avoids overrunning the allotted
	 * time slice.
	 */
	if (sample_valid(cic->ttime.ttime_samples) &&
	    (blurq->slice_end - jiffies < cic->ttime.ttime_mean)) {
		blur_log_blurq(blurd, blurq, "Not idling. think_time:%lu",
			     cic->ttime.ttime_mean);
		return;
	}

	/* There are other queues in the group, don't do group idle */
	if (group_idle && blurq->blurg->nr_blurq > 1)
		return;

	blur_mark_blurq_wait_request(blurq);

	if (group_idle)
		sl = blurd->blur_group_idle;
	else
		sl = blurd->blur_slice_idle;

	mod_timer(&blurd->idle_slice_timer, jiffies + sl);
	blur_blkiocg_update_set_idle_time_stats(&blurq->blurg->blkg);
	blur_log_blurq(blurd, blurq, "arm_idle: %lu group_idle: %d", sl,
			group_idle ? 1 : 0);
}

/*
 * Move request from internal lists to the request queue dispatch list.
 */
static void blur_dispatch_insert(struct request_queue *q, struct request *rq)
{
	struct blur_data *blurd = q->elevator->elevator_data;
	struct blur_queue *blurq = RQ_blurQ(rq);

	blur_log_blurq(blurd, blurq, "dispatch_insert");

	blurq->next_rq = blur_find_next_rq(blurd, blurq, rq);
	blur_remove_request(rq);
	blurq->dispatched++;
	(RQ_blurG(rq))->dispatched++;
	elv_dispatch_sort(q, rq);

	blurd->rq_in_flight[blur_blurq_sync(blurq)]++;
	blurq->nr_sectors += blk_rq_sectors(rq);
	blur_blkiocg_update_dispatch_stats(&blurq->blurg->blkg, blk_rq_bytes(rq),
					rq_data_dir(rq), rq_is_sync(rq));
}

/*
 * return expired entry, or NULL to just start from scratch in rbtree
 */
static struct request *blur_check_fifo(struct blur_queue *blurq)
{
	struct request *rq = NULL;

	if (blur_blurq_fifo_expire(blurq))
		return NULL;

	blur_mark_blurq_fifo_expire(blurq);

	if (list_empty(&blurq->fifo))
		return NULL;

	rq = rq_entry_fifo(blurq->fifo.next);
	if (time_before(jiffies, rq_fifo_time(rq)))
		rq = NULL;

	blur_log_blurq(blurq->blurd, blurq, "fifo=%p", rq);
	return rq;
}

static inline int
blur_prio_to_maxrq(struct blur_data *blurd, struct blur_queue *blurq)
{
	const int base_rq = blurd->blur_slice_async_rq;

	WARN_ON(blurq->ioprio >= IOPRIO_BE_NR);

	return 2 * base_rq * (IOPRIO_BE_NR - blurq->ioprio);
}

/*
 * Must be called with the queue_lock held.
 */
static int blurq_process_refs(struct blur_queue *blurq)
{
	int process_refs, io_refs;

	io_refs = blurq->allocated[READ] + blurq->allocated[WRITE];
	process_refs = blurq->ref - io_refs;
	BUG_ON(process_refs < 0);
	return process_refs;
}

static void blur_setup_merge(struct blur_queue *blurq, struct blur_queue *new_blurq)
{
	int process_refs, new_process_refs;
	struct blur_queue *__blurq;

	/*
	 * If there are no process references on the new_blurq, then it is
	 * unsafe to follow the ->new_blurq chain as other blurq's in the
	 * chain may have dropped their last reference (not just their
	 * last process reference).
	 */
	if (!blurq_process_refs(new_blurq))
		return;

	/* Avoid a circular list and skip interim queue merges */
	while ((__blurq = new_blurq->new_blurq)) {
		if (__blurq == blurq)
			return;
		new_blurq = __blurq;
	}

	process_refs = blurq_process_refs(blurq);
	new_process_refs = blurq_process_refs(new_blurq);
	/*
	 * If the process for the blurq has gone away, there is no
	 * sense in merging the queues.
	 */
	if (process_refs == 0 || new_process_refs == 0)
		return;

	/*
	 * Merge in the direction of the lesser amount of work.
	 */
	if (new_process_refs >= process_refs) {
		blurq->new_blurq = new_blurq;
		new_blurq->ref += process_refs;
	} else {
		new_blurq->new_blurq = blurq;
		blurq->ref += new_process_refs;
	}
}

static enum wl_type_t blur_choose_wl(struct blur_data *blurd,
				struct blur_group *blurg, enum wl_prio_t prio)
{
	struct blur_queue *queue;
	int i;
	bool key_valid = false;
	unsigned long lowest_key = 0;
	enum wl_type_t cur_best = SYNC_NOIDLE_WORKLOAD;

	for (i = 0; i <= SYNC_WORKLOAD; ++i) {
		/* select the one with lowest rb_key */
		queue = blur_rb_first(service_tree_for(blurg, prio, i));
		if (queue &&
		    (!key_valid || time_before(queue->rb_key, lowest_key))) {
			lowest_key = queue->rb_key;
			cur_best = i;
			key_valid = true;
		}
	}

	return cur_best;
}

static void choose_service_tree(struct blur_data *blurd, struct blur_group *blurg)
{
	unsigned slice;
	unsigned count;
	struct blur_rb_root *st;
	unsigned group_slice;
	enum wl_prio_t original_prio = blurd->serving_prio;

	/* Choose next priority. RT > BE > IDLE */
	if (blur_group_busy_queues_wl(RT_WORKLOAD, blurd, blurg))
		blurd->serving_prio = RT_WORKLOAD;
	else if (blur_group_busy_queues_wl(BE_WORKLOAD, blurd, blurg))
		blurd->serving_prio = BE_WORKLOAD;
	else {
		blurd->serving_prio = IDLE_WORKLOAD;
		blurd->workload_expires = jiffies + 1;
		return;
	}

	if (original_prio != blurd->serving_prio)
		goto new_workload;

	/*
	 * For RT and BE, we have to choose also the type
	 * (SYNC, SYNC_NOIDLE, ASYNC), and to compute a workload
	 * expiration time
	 */
	st = service_tree_for(blurg, blurd->serving_prio, blurd->serving_type);
	count = st->count;

	/*
	 * check workload expiration, and that we still have other queues ready
	 */
	if (count && !time_after(jiffies, blurd->workload_expires))
		return;

new_workload:
	/* otherwise select new workload type */
	blurd->serving_type =
		blur_choose_wl(blurd, blurg, blurd->serving_prio);
	st = service_tree_for(blurg, blurd->serving_prio, blurd->serving_type);
	count = st->count;

	/*
	 * the workload slice is computed as a fraction of target latency
	 * proportional to the number of queues in that workload, over
	 * all the queues in the same priority class
	 */
	group_slice = blur_group_slice(blurd, blurg);

	slice = group_slice * count /
		max_t(unsigned, blurg->busy_queues_avg[blurd->serving_prio],
		      blur_group_busy_queues_wl(blurd->serving_prio, blurd, blurg));

	if (blurd->serving_type == ASYNC_WORKLOAD) {
		unsigned int tmp;

		/*
		 * Async queues are currently system wide. Just taking
		 * proportion of queues with-in same group will lead to higher
		 * async ratio system wide as generally root group is going
		 * to have higher weight. A more accurate thing would be to
		 * calculate system wide asnc/sync ratio.
		 */
		tmp = blurd->blur_target_latency *
			blurg_busy_async_queues(blurd, blurg);
		tmp = tmp/blurd->busy_queues;
		slice = min_t(unsigned, slice, tmp);

		/* async workload slice is scaled down according to
		 * the sync/async slice ratio. */
		slice = slice * blurd->blur_slice[0] / blurd->blur_slice[1];
	} else
		/* sync workload slice is at least 2 * blur_slice_idle */
		slice = max(slice, 2 * blurd->blur_slice_idle);

	slice = max_t(unsigned, slice, blur_MIN_TT);
	blur_log(blurd, "workload slice:%d", slice);
	blurd->workload_expires = jiffies + slice;
}

static struct blur_group *blur_get_next_blurg(struct blur_data *blurd)
{
	struct blur_rb_root *st = &blurd->grp_service_tree;
	struct blur_group *blurg;

	if (RB_EMPTY_ROOT(&st->rb))
		return NULL;
	blurg = blur_rb_first_group(st);
	update_min_vdisktime(st);
	return blurg;
}

static void blur_choose_blurg(struct blur_data *blurd)
{
	struct blur_group *blurg = blur_get_next_blurg(blurd);

	if (!blurg)
		return;

	blurd->serving_group = blurg;

	/* Restore the workload type data */
	if (blurg->saved_workload_slice) {
		blurd->workload_expires = jiffies + blurg->saved_workload_slice;
		blurd->serving_type = blurg->saved_workload;
		blurd->serving_prio = blurg->saved_serving_prio;
	} else
		blurd->workload_expires = jiffies - 1;

	choose_service_tree(blurd, blurg);
}

/*
 * Select a queue for service. If we have a current active queue,
 * check whether to continue servicing it, or retrieve and set a new one.
 */
static struct blur_queue *blur_select_queue(struct blur_data *blurd)
{
	struct blur_queue *blurq, *new_blurq = NULL;

	blurq = blurd->active_queue;
	if (!blurq)
		goto new_queue;

	if (!blurd->rq_queued)
		return NULL;

	/*
	 * We were waiting for group to get backlogged. Expire the queue
	 */
	if (blur_blurq_wait_busy(blurq) && !RB_EMPTY_ROOT(&blurq->sort_list))
		goto expire;

	/*
	 * The active queue has run out of time, expire it and select new.
	 */
	if (blur_slice_used(blurq) && !blur_blurq_must_dispatch(blurq)) {
		/*
		 * If slice had not expired at the completion of last request
		 * we might not have turned on wait_busy flag. Don't expire
		 * the queue yet. Allow the group to get backlogged.
		 *
		 * The very fact that we have used the slice, that means we
		 * have been idling all along on this queue and it should be
		 * ok to wait for this request to complete.
		 */
		if (blurq->blurg->nr_blurq == 1 && RB_EMPTY_ROOT(&blurq->sort_list)
		    && blurq->dispatched && blur_should_idle(blurd, blurq)) {
			blurq = NULL;
			goto keep_queue;
		} else
			goto check_group_idle;
	}

	/*
	 * The active queue has requests and isn't expired, allow it to
	 * dispatch.
	 */
	if (!RB_EMPTY_ROOT(&blurq->sort_list))
		goto keep_queue;

	/*
	 * If another queue has a request waiting within our mean seek
	 * distance, let it run.  The expire code will check for close
	 * cooperators and put the close queue at the front of the service
	 * tree.  If possible, merge the expiring queue with the new blurq.
	 */
	new_blurq = blur_close_cooperator(blurd, blurq);
	if (new_blurq) {
		if (!blurq->new_blurq)
			blur_setup_merge(blurq, new_blurq);
		goto expire;
	}

	/*
	 * No requests pending. If the active queue still has requests in
	 * flight or is idling for a new request, allow either of these
	 * conditions to happen (or time out) before selecting a new queue.
	 */
	if (timer_pending(&blurd->idle_slice_timer)) {
		blurq = NULL;
		goto keep_queue;
	}

	/*
	 * This is a deep seek queue, but the device is much faster than
	 * the queue can deliver, don't idle
	 **/
	if (blurQ_SEEKY(blurq) && blur_blurq_idle_window(blurq) &&
	    (blur_blurq_slice_new(blurq) ||
	    (blurq->slice_end - jiffies > jiffies - blurq->slice_start))) {
		blur_clear_blurq_deep(blurq);
		blur_clear_blurq_idle_window(blurq);
	}

	if (blurq->dispatched && blur_should_idle(blurd, blurq)) {
		blurq = NULL;
		goto keep_queue;
	}

	/*
	 * If group idle is enabled and there are requests dispatched from
	 * this group, wait for requests to complete.
	 */
check_group_idle:
	if (blurd->blur_group_idle && blurq->blurg->nr_blurq == 1 &&
	    blurq->blurg->dispatched &&
	    !blur_io_thinktime_big(blurd, &blurq->blurg->ttime, true)) {
		blurq = NULL;
		goto keep_queue;
	}

expire:
	blur_slice_expired(blurd, 0);
new_queue:
	/*
	 * Current queue expired. Check if we have to switch to a new
	 * service tree
	 */
	if (!new_blurq)
		blur_choose_blurg(blurd);

	blurq = blur_set_active_queue(blurd, new_blurq);
keep_queue:
	return blurq;
}

static int __blur_forced_dispatch_blurq(struct blur_queue *blurq)
{
	int dispatched = 0;

	while (blurq->next_rq) {
		blur_dispatch_insert(blurq->blurd->queue, blurq->next_rq);
		dispatched++;
	}

	BUG_ON(!list_empty(&blurq->fifo));

	/* By default blurq is not expired if it is empty. Do it explicitly */
	__blur_slice_expired(blurq->blurd, blurq, 0);
	return dispatched;
}

/*
 * Drain our current requests. Used for barriers and when switching
 * io schedulers on-the-fly.
 */
static int blur_forced_dispatch(struct blur_data *blurd)
{
	struct blur_queue *blurq;
	int dispatched = 0;

	/* Expire the timeslice of the current active queue first */
	blur_slice_expired(blurd, 0);
	while ((blurq = blur_get_next_queue_forced(blurd)) != NULL) {
		__blur_set_active_queue(blurd, blurq);
		dispatched += __blur_forced_dispatch_blurq(blurq);
	}

	BUG_ON(blurd->busy_queues);

	blur_log(blurd, "forced_dispatch=%d", dispatched);
	return dispatched;
}

static inline bool blur_slice_used_soon(struct blur_data *blurd,
	struct blur_queue *blurq)
{
	/* the queue hasn't finished any request, can't estimate */
	if (blur_blurq_slice_new(blurq))
		return true;
	if (time_after(jiffies + blurd->blur_slice_idle * blurq->dispatched,
		blurq->slice_end))
		return true;

	return false;
}

static bool blur_may_dispatch(struct blur_data *blurd, struct blur_queue *blurq)
{
	unsigned int max_dispatch;

	/*
	 * Drain async requests before we start sync IO
	 */
	if (blur_should_idle(blurd, blurq) && blurd->rq_in_flight[BLK_RW_ASYNC])
		return false;

	/*
	 * If this is an async queue and we have sync IO in flight, let it wait
	 */
	if (blurd->rq_in_flight[BLK_RW_SYNC] && !blur_blurq_sync(blurq))
		return false;

	max_dispatch = max_t(unsigned int, blurd->blur_quantum / 2, 1);
	if (blur_class_idle(blurq))
		max_dispatch = 1;

	/*
	 * Does this blurq already have too much IO in flight?
	 */
	if (blurq->dispatched >= max_dispatch) {
		bool promote_sync = false;
		/*
		 * idle queue must always only have a single IO in flight
		 */
		if (blur_class_idle(blurq))
			return false;

		/*
		 * If there is only one sync queue
		 * we can ignore async queue here and give the sync
		 * queue no dispatch limit. The reason is a sync queue can
		 * preempt async queue, limiting the sync queue doesn't make
		 * sense. This is useful for aiostress test.
		 */
		if (blur_blurq_sync(blurq) && blurd->busy_sync_queues == 1)
			promote_sync = true;

		/*
		 * We have other queues, don't allow more IO from this one
		 */
		if (blurd->busy_queues > 1 && blur_slice_used_soon(blurd, blurq) &&
				!promote_sync)
			return false;

		/*
		 * Sole queue user, no limit
		 */
		if (blurd->busy_queues == 1 || promote_sync)
			max_dispatch = -1;
		else
			/*
			 * Normally we start throttling blurq when blur_quantum/2
			 * requests have been dispatched. But we can drive
			 * deeper queue depths at the beginning of slice
			 * subjected to upper limit of blur_quantum.
			 * */
			max_dispatch = blurd->blur_quantum;
	}

	/*
	 * Async queues must wait a bit before being allowed dispatch.
	 * We also ramp up the dispatch depth gradually for async IO,
	 * based on the last sync IO we serviced
	 */
	if (!blur_blurq_sync(blurq) && blurd->blur_latency) {
		unsigned long last_sync = jiffies - blurd->last_delayed_sync;
		unsigned int depth;

		depth = last_sync / blurd->blur_slice[1];
		if (!depth && !blurq->dispatched)
			depth = 1;
		if (depth < max_dispatch)
			max_dispatch = depth;
	}

	/*
	 * If we're below the current max, allow a dispatch
	 */
	return blurq->dispatched < max_dispatch;
}

/*
 * Dispatch a request from blurq, moving them to the request queue
 * dispatch list.
 */
static bool blur_dispatch_request(struct blur_data *blurd, struct blur_queue *blurq)
{
	struct request *rq;

	BUG_ON(RB_EMPTY_ROOT(&blurq->sort_list));

	if (!blur_may_dispatch(blurd, blurq))
		return false;

	/*
	 * follow expired path, else get first next available
	 */
	rq = blur_check_fifo(blurq);
	if (!rq)
		rq = blurq->next_rq;

	/*
	 * insert request into driver dispatch list
	 */
	blur_dispatch_insert(blurd->queue, rq);

	if (!blurd->active_cic) {
		struct blur_io_cq *cic = RQ_CIC(rq);

		atomic_long_inc(&cic->icq.ioc->refcount);
		blurd->active_cic = cic;
	}

	return true;
}

/*
 * Find the blurq that we need to service and move a request from that to the
 * dispatch list
 */
static int blur_dispatch_requests(struct request_queue *q, int force)
{
	struct blur_data *blurd = q->elevator->elevator_data;
	struct blur_queue *blurq;

	if (!blurd->busy_queues)
		return 0;

	if (unlikely(force))
		return blur_forced_dispatch(blurd);

	blurq = blur_select_queue(blurd);
	if (!blurq)
		return 0;

	/*
	 * Dispatch a request from this blurq, if it is allowed
	 */
	if (!blur_dispatch_request(blurd, blurq))
		return 0;

	blurq->slice_dispatch++;
	blur_clear_blurq_must_dispatch(blurq);

	/*
	 * expire an async queue immediately if it has used up its slice. idle
	 * queue always expire after 1 dispatch round.
	 */
	if (blurd->busy_queues > 1 && ((!blur_blurq_sync(blurq) &&
	    blurq->slice_dispatch >= blur_prio_to_maxrq(blurd, blurq)) ||
	    blur_class_idle(blurq))) {
		blurq->slice_end = jiffies + 1;
		blur_slice_expired(blurd, 0);
	}

	blur_log_blurq(blurd, blurq, "dispatched a request");
	return 1;
}

/*
 * task holds one reference to the queue, dropped when task exits. each rq
 * in-flight on this queue also holds a reference, dropped when rq is freed.
 *
 * Each blur queue took a reference on the parent group. Drop it now.
 * queue lock must be held here.
 */
static void blur_put_queue(struct blur_queue *blurq)
{
	struct blur_data *blurd = blurq->blurd;
	struct blur_group *blurg;

	BUG_ON(blurq->ref <= 0);

	blurq->ref--;
	if (blurq->ref)
		return;

	blur_log_blurq(blurd, blurq, "put_queue");
	BUG_ON(rb_first(&blurq->sort_list));
	BUG_ON(blurq->allocated[READ] + blurq->allocated[WRITE]);
	blurg = blurq->blurg;

	if (unlikely(blurd->active_queue == blurq)) {
		__blur_slice_expired(blurd, blurq, 0);
		blur_schedule_dispatch(blurd);
	}

	BUG_ON(blur_blurq_on_rr(blurq));
	kmem_cache_free(blur_pool, blurq);
	blur_put_blurg(blurg);
}

static void blur_put_cooperator(struct blur_queue *blurq)
{
	struct blur_queue *__blurq, *next;

	/*
	 * If this queue was scheduled to merge with another queue, be
	 * sure to drop the reference taken on that queue (and others in
	 * the merge chain).  See blur_setup_merge and blur_merge_blurqs.
	 */
	__blurq = blurq->new_blurq;
	while (__blurq) {
		if (__blurq == blurq) {
			WARN(1, "blurq->new_blurq loop detected\n");
			break;
		}
		next = __blurq->new_blurq;
		blur_put_queue(__blurq);
		__blurq = next;
	}
}

static void blur_exit_blurq(struct blur_data *blurd, struct blur_queue *blurq)
{
	if (unlikely(blurq == blurd->active_queue)) {
		__blur_slice_expired(blurd, blurq, 0);
		blur_schedule_dispatch(blurd);
	}

	blur_put_cooperator(blurq);

	blur_put_queue(blurq);
}

static void blur_init_icq(struct io_cq *icq)
{
	struct blur_io_cq *cic = icq_to_cic(icq);

	cic->ttime.last_end_request = jiffies;
}

static void blur_exit_icq(struct io_cq *icq)
{
	struct blur_io_cq *cic = icq_to_cic(icq);
	struct blur_data *blurd = cic_to_blurd(cic);

	if (cic->blurq[BLK_RW_ASYNC]) {
		blur_exit_blurq(blurd, cic->blurq[BLK_RW_ASYNC]);
		cic->blurq[BLK_RW_ASYNC] = NULL;
	}

	if (cic->blurq[BLK_RW_SYNC]) {
		blur_exit_blurq(blurd, cic->blurq[BLK_RW_SYNC]);
		cic->blurq[BLK_RW_SYNC] = NULL;
	}
}

static void blur_init_prio_data(struct blur_queue *blurq, struct io_context *ioc)
{
	struct task_struct *tsk = current;
	int ioprio_class;

	if (!blur_blurq_prio_changed(blurq))
		return;

	ioprio_class = IOPRIO_PRIO_CLASS(ioc->ioprio);
	switch (ioprio_class) {
	default:
		printk(KERN_ERR "blur: bad prio %x\n", ioprio_class);
	case IOPRIO_CLASS_NONE:
		/*
		 * no prio set, inherit CPU scheduling settings
		 */
		blurq->ioprio = task_nice_ioprio(tsk);
		blurq->ioprio_class = task_nice_ioclass(tsk);
		break;
	case IOPRIO_CLASS_RT:
		blurq->ioprio = task_ioprio(ioc);
		blurq->ioprio_class = IOPRIO_CLASS_RT;
		break;
	case IOPRIO_CLASS_BE:
		blurq->ioprio = task_ioprio(ioc);
		blurq->ioprio_class = IOPRIO_CLASS_BE;
		break;
	case IOPRIO_CLASS_IDLE:
		blurq->ioprio_class = IOPRIO_CLASS_IDLE;
		blurq->ioprio = 7;
		blur_clear_blurq_idle_window(blurq);
		break;
	}

	/*
	 * keep track of original prio settings in case we have to temporarily
	 * elevate the priority of this queue
	 */
	blurq->org_ioprio = blurq->ioprio;
	blur_clear_blurq_prio_changed(blurq);
}

static void changed_ioprio(struct blur_io_cq *cic)
{
	struct blur_data *blurd = cic_to_blurd(cic);
	struct blur_queue *blurq;

	if (unlikely(!blurd))
		return;

	blurq = cic->blurq[BLK_RW_ASYNC];
	if (blurq) {
		struct blur_queue *new_blurq;
		new_blurq = blur_get_queue(blurd, BLK_RW_ASYNC, cic->icq.ioc,
						GFP_ATOMIC);
		if (new_blurq) {
			cic->blurq[BLK_RW_ASYNC] = new_blurq;
			blur_put_queue(blurq);
		}
	}

	blurq = cic->blurq[BLK_RW_SYNC];
	if (blurq)
		blur_mark_blurq_prio_changed(blurq);
}

static void blur_init_blurq(struct blur_data *blurd, struct blur_queue *blurq,
			  pid_t pid, bool is_sync)
{
	RB_CLEAR_NODE(&blurq->rb_node);
	RB_CLEAR_NODE(&blurq->p_node);
	INIT_LIST_HEAD(&blurq->fifo);

	blurq->ref = 0;
	blurq->blurd = blurd;

	blur_mark_blurq_prio_changed(blurq);

	if (is_sync) {
		if (!blur_class_idle(blurq))
			blur_mark_blurq_idle_window(blurq);
		blur_mark_blurq_sync(blurq);
	}
	blurq->pid = pid;
}

#ifdef CONFIG_blur_GROUP_IOSCHED
static void changed_cgroup(struct blur_io_cq *cic)
{
	struct blur_queue *sync_blurq = cic_to_blurq(cic, 1);
	struct blur_data *blurd = cic_to_blurd(cic);
	struct request_queue *q;

	if (unlikely(!blurd))
		return;

	q = blurd->queue;

	if (sync_blurq) {
		/*
		 * Drop reference to sync queue. A new sync queue will be
		 * assigned in new group upon arrival of a fresh request.
		 */
		blur_log_blurq(blurd, sync_blurq, "changed cgroup");
		cic_set_blurq(cic, NULL, 1);
		blur_put_queue(sync_blurq);
	}
}
#endif  /* CONFIG_blur_GROUP_IOSCHED */

static struct blur_queue *
blur_find_alloc_queue(struct blur_data *blurd, bool is_sync,
		     struct io_context *ioc, gfp_t gfp_mask)
{
	struct blur_queue *blurq, *new_blurq = NULL;
	struct blur_io_cq *cic;
	struct blur_group *blurg;

retry:
	blurg = blur_get_blurg(blurd);
	cic = blur_cic_lookup(blurd, ioc);
	/* cic always exists here */
	blurq = cic_to_blurq(cic, is_sync);

	/*
	 * Always try a new alloc if we fell back to the OOM blurq
	 * originally, since it should just be a temporary situation.
	 */
	if (!blurq || blurq == &blurd->oom_blurq) {
		blurq = NULL;
		if (new_blurq) {
			blurq = new_blurq;
			new_blurq = NULL;
		} else if (gfp_mask & __GFP_WAIT) {
			spin_unlock_irq(blurd->queue->queue_lock);
			new_blurq = kmem_cache_alloc_node(blur_pool,
					gfp_mask | __GFP_ZERO,
					blurd->queue->node);
			spin_lock_irq(blurd->queue->queue_lock);
			if (new_blurq)
				goto retry;
		} else {
			blurq = kmem_cache_alloc_node(blur_pool,
					gfp_mask | __GFP_ZERO,
					blurd->queue->node);
		}

		if (blurq) {
			blur_init_blurq(blurd, blurq, current->pid, is_sync);
			blur_init_prio_data(blurq, ioc);
			blur_link_blurq_blurg(blurq, blurg);
			blur_log_blurq(blurd, blurq, "alloced");
		} else
			blurq = &blurd->oom_blurq;
	}

	if (new_blurq)
		kmem_cache_free(blur_pool, new_blurq);

	return blurq;
}

static struct blur_queue **
blur_async_queue_prio(struct blur_data *blurd, int ioprio_class, int ioprio)
{
	switch (ioprio_class) {
	case IOPRIO_CLASS_RT:
		return &blurd->async_blurq[0][ioprio];
	case IOPRIO_CLASS_BE:
		return &blurd->async_blurq[1][ioprio];
	case IOPRIO_CLASS_IDLE:
		return &blurd->async_idle_blurq;
	default:
		BUG();
	}
}

static struct blur_queue *
blur_get_queue(struct blur_data *blurd, bool is_sync, struct io_context *ioc,
	      gfp_t gfp_mask)
{
	const int ioprio = task_ioprio(ioc);
	const int ioprio_class = task_ioprio_class(ioc);
	struct blur_queue **async_blurq = NULL;
	struct blur_queue *blurq = NULL;

	if (!is_sync) {
		async_blurq = blur_async_queue_prio(blurd, ioprio_class, ioprio);
		blurq = *async_blurq;
	}

	if (!blurq)
		blurq = blur_find_alloc_queue(blurd, is_sync, ioc, gfp_mask);

	/*
	 * pin the queue now that it's allocated, scheduler exit will prune it
	 */
	if (!is_sync && !(*async_blurq)) {
		blurq->ref++;
		*async_blurq = blurq;
	}

	blurq->ref++;
	return blurq;
}

static void
__blur_update_io_thinktime(struct blur_ttime *ttime, unsigned long slice_idle)
{
	unsigned long elapsed = jiffies - ttime->last_end_request;
	elapsed = min(elapsed, 2UL * slice_idle);

	ttime->ttime_samples = (7*ttime->ttime_samples + 256) / 8;
	ttime->ttime_total = (7*ttime->ttime_total + 256*elapsed) / 8;
	ttime->ttime_mean = (ttime->ttime_total + 128) / ttime->ttime_samples;
}

static void
blur_update_io_thinktime(struct blur_data *blurd, struct blur_queue *blurq,
			struct blur_io_cq *cic)
{
	if (blur_blurq_sync(blurq)) {
		__blur_update_io_thinktime(&cic->ttime, blurd->blur_slice_idle);
		__blur_update_io_thinktime(&blurq->service_tree->ttime,
			blurd->blur_slice_idle);
	}
#ifdef CONFIG_blur_GROUP_IOSCHED
	__blur_update_io_thinktime(&blurq->blurg->ttime, blurd->blur_group_idle);
#endif
}

static void
blur_update_io_seektime(struct blur_data *blurd, struct blur_queue *blurq,
		       struct request *rq)
{
	sector_t sdist = 0;
	sector_t n_sec = blk_rq_sectors(rq);
	if (blurq->last_request_pos) {
		if (blurq->last_request_pos < blk_rq_pos(rq))
			sdist = blk_rq_pos(rq) - blurq->last_request_pos;
		else
			sdist = blurq->last_request_pos - blk_rq_pos(rq);
	}

	blurq->seek_history <<= 1;
	if (blk_queue_nonrot(blurd->queue))
		blurq->seek_history |= (n_sec < blurQ_SECT_THR_NONROT);
	else
		blurq->seek_history |= (sdist > blurQ_SEEK_THR);
}

/*
 * Disable idle window if the process thinks too long or seeks so much that
 * it doesn't matter
 */
static void
blur_update_idle_window(struct blur_data *blurd, struct blur_queue *blurq,
		       struct blur_io_cq *cic)
{
	int old_idle, enable_idle;

	/*
	 * Don't idle for async or idle io prio class
	 */
	if (!blur_blurq_sync(blurq) || blur_class_idle(blurq))
		return;

	enable_idle = old_idle = blur_blurq_idle_window(blurq);

	if (blurq->queued[0] + blurq->queued[1] >= 4)
		blur_mark_blurq_deep(blurq);

	if (blurq->next_rq && (blurq->next_rq->cmd_flags & REQ_NOIDLE))
		enable_idle = 0;
	else if (!atomic_read(&cic->icq.ioc->nr_tasks) ||
		 !blurd->blur_slice_idle ||
		 (!blur_blurq_deep(blurq) && blurQ_SEEKY(blurq)))
		enable_idle = 0;
	else if (sample_valid(cic->ttime.ttime_samples)) {
		if (cic->ttime.ttime_mean > blurd->blur_slice_idle)
			enable_idle = 0;
		else
			enable_idle = 1;
	}

	if (old_idle != enable_idle) {
		blur_log_blurq(blurd, blurq, "idle=%d", enable_idle);
		if (enable_idle)
			blur_mark_blurq_idle_window(blurq);
		else
			blur_clear_blurq_idle_window(blurq);
	}
}

/*
 * Check if new_blurq should preempt the currently active queue. Return 0 for
 * no or if we aren't sure, a 1 will cause a preempt.
 */
static bool
blur_should_preempt(struct blur_data *blurd, struct blur_queue *new_blurq,
		   struct request *rq)
{
	struct blur_queue *blurq;

	blurq = blurd->active_queue;
	if (!blurq)
		return false;

	if (blur_class_idle(new_blurq))
		return false;

	if (blur_class_idle(blurq))
		return true;

	/*
	 * Don't allow a non-RT request to preempt an ongoing RT blurq timeslice.
	 */
	if (blur_class_rt(blurq) && !blur_class_rt(new_blurq))
		return false;

	/*
	 * if the new request is sync, but the currently running queue is
	 * not, let the sync request have priority.
	 */
	if (rq_is_sync(rq) && !blur_blurq_sync(blurq))
		return true;

	if (new_blurq->blurg != blurq->blurg)
		return false;

	if (blur_slice_used(blurq))
		return true;

	/* Allow preemption only if we are idling on sync-noidle tree */
	if (blurd->serving_type == SYNC_NOIDLE_WORKLOAD &&
	    blurq_type(new_blurq) == SYNC_NOIDLE_WORKLOAD &&
	    new_blurq->service_tree->count == 2 &&
	    RB_EMPTY_ROOT(&blurq->sort_list))
		return true;

	/*
	 * So both queues are sync. Let the new request get disk time if
	 * it's a metadata request and the current queue is doing regular IO.
	 */
	if ((rq->cmd_flags & REQ_PRIO) && !blurq->prio_pending)
		return true;

	/*
	 * Allow an RT request to pre-empt an ongoing non-RT blurq timeslice.
	 */
	if (blur_class_rt(new_blurq) && !blur_class_rt(blurq))
		return true;

	/* An idle queue should not be idle now for some reason */
	if (RB_EMPTY_ROOT(&blurq->sort_list) && !blur_should_idle(blurd, blurq))
		return true;

	if (!blurd->active_cic || !blur_blurq_wait_request(blurq))
		return false;

	/*
	 * if this request is as-good as one we would expect from the
	 * current blurq, let it preempt
	 */
	if (blur_rq_close(blurd, blurq, rq))
		return true;

	return false;
}

/*
 * blurq preempts the active queue. if we allowed preempt with no slice left,
 * let it have half of its nominal slice.
 */
static void blur_preempt_queue(struct blur_data *blurd, struct blur_queue *blurq)
{
	enum wl_type_t old_type = blurq_type(blurd->active_queue);

	blur_log_blurq(blurd, blurq, "preempt");
	blur_slice_expired(blurd, 1);

	/*
	 * workload type is changed, don't save slice, otherwise preempt
	 * doesn't happen
	 */
	if (old_type != blurq_type(blurq))
		blurq->blurg->saved_workload_slice = 0;

	/*
	 * Put the new queue at the front of the of the current list,
	 * so we know that it will be selected next.
	 */
	BUG_ON(!blur_blurq_on_rr(blurq));

	blur_service_tree_add(blurd, blurq, 1);

	blurq->slice_end = 0;
	blur_mark_blurq_slice_new(blurq);
}

/*
 * Called when a new fs request (rq) is added (to blurq). Check if there's
 * something we should do about it
 */
static void
blur_rq_enqueued(struct blur_data *blurd, struct blur_queue *blurq,
		struct request *rq)
{
	struct blur_io_cq *cic = RQ_CIC(rq);

	blurd->rq_queued++;
	if (rq->cmd_flags & REQ_PRIO)
		blurq->prio_pending++;

	blur_update_io_thinktime(blurd, blurq, cic);
	blur_update_io_seektime(blurd, blurq, rq);
	blur_update_idle_window(blurd, blurq, cic);

	blurq->last_request_pos = blk_rq_pos(rq) + blk_rq_sectors(rq);

	if (blurq == blurd->active_queue) {
		/*
		 * Remember that we saw a request from this process, but
		 * don't start queuing just yet. Otherwise we risk seeing lots
		 * of tiny requests, because we disrupt the normal plugging
		 * and merging. If the request is already larger than a single
		 * page, let it rip immediately. For that case we assume that
		 * merging is already done. Ditto for a busy system that
		 * has other work pending, don't risk delaying until the
		 * idle timer unplug to continue working.
		 */
		if (blur_blurq_wait_request(blurq)) {
			if (blk_rq_bytes(rq) > PAGE_CACHE_SIZE ||
			    blurd->busy_queues > 1) {
				blur_del_timer(blurd, blurq);
				blur_clear_blurq_wait_request(blurq);
				__blk_run_queue(blurd->queue);
			} else {
				blur_blkiocg_update_idle_time_stats(
						&blurq->blurg->blkg);
				blur_mark_blurq_must_dispatch(blurq);
			}
		}
	} else if (blur_should_preempt(blurd, blurq, rq)) {
		/*
		 * not the active queue - expire current slice if it is
		 * idle and has expired it's mean thinktime or this new queue
		 * has some old slice time left and is of higher priority or
		 * this new queue is RT and the current one is BE
		 */
		blur_preempt_queue(blurd, blurq);
		__blk_run_queue(blurd->queue);
	}
}

static void blur_insert_request(struct request_queue *q, struct request *rq)
{
	struct blur_data *blurd = q->elevator->elevator_data;
	struct blur_queue *blurq = RQ_blurQ(rq);

	blur_log_blurq(blurd, blurq, "insert_request");
	blur_init_prio_data(blurq, RQ_CIC(rq)->icq.ioc);

	rq_set_fifo_time(rq, jiffies + blurd->blur_fifo_expire[rq_is_sync(rq)]);
	list_add_tail(&rq->queuelist, &blurq->fifo);
	blur_add_rq_rb(rq);
	blur_blkiocg_update_io_add_stats(&(RQ_blurG(rq))->blkg,
			&blurd->serving_group->blkg, rq_data_dir(rq),
			rq_is_sync(rq));
	blur_rq_enqueued(blurd, blurq, rq);
}

/*
 * Update hw_tag based on peak queue depth over 50 samples under
 * sufficient load.
 */
static void blur_update_hw_tag(struct blur_data *blurd)
{
	struct blur_queue *blurq = blurd->active_queue;

	if (blurd->rq_in_driver > blurd->hw_tag_est_depth)
		blurd->hw_tag_est_depth = blurd->rq_in_driver;

	if (blurd->hw_tag == 1)
		return;

	if (blurd->rq_queued <= blur_HW_QUEUE_MIN &&
	    blurd->rq_in_driver <= blur_HW_QUEUE_MIN)
		return;

	/*
	 * If active queue hasn't enough requests and can idle, blur might not
	 * dispatch sufficient requests to hardware. Don't zero hw_tag in this
	 * case
	 */
	if (blurq && blur_blurq_idle_window(blurq) &&
	    blurq->dispatched + blurq->queued[0] + blurq->queued[1] <
	    blur_HW_QUEUE_MIN && blurd->rq_in_driver < blur_HW_QUEUE_MIN)
		return;

	if (blurd->hw_tag_samples++ < 50)
		return;

	if (blurd->hw_tag_est_depth >= blur_HW_QUEUE_MIN)
		blurd->hw_tag = 1;
	else
		blurd->hw_tag = 0;
}

static bool blur_should_wait_busy(struct blur_data *blurd, struct blur_queue *blurq)
{
	struct blur_io_cq *cic = blurd->active_cic;

	/* If the queue already has requests, don't wait */
	if (!RB_EMPTY_ROOT(&blurq->sort_list))
		return false;

	/* If there are other queues in the group, don't wait */
	if (blurq->blurg->nr_blurq > 1)
		return false;

	/* the only queue in the group, but think time is big */
	if (blur_io_thinktime_big(blurd, &blurq->blurg->ttime, true))
		return false;

	if (blur_slice_used(blurq))
		return true;

	/* if slice left is less than think time, wait busy */
	if (cic && sample_valid(cic->ttime.ttime_samples)
	    && (blurq->slice_end - jiffies < cic->ttime.ttime_mean))
		return true;

	/*
	 * If think times is less than a jiffy than ttime_mean=0 and above
	 * will not be true. It might happen that slice has not expired yet
	 * but will expire soon (4-5 ns) during select_queue(). To cover the
	 * case where think time is less than a jiffy, mark the queue wait
	 * busy if only 1 jiffy is left in the slice.
	 */
	if (blurq->slice_end - jiffies == 1)
		return true;

	return false;
}

static void blur_completed_request(struct request_queue *q, struct request *rq)
{
	struct blur_queue *blurq = RQ_blurQ(rq);
	struct blur_data *blurd = blurq->blurd;
	const int sync = rq_is_sync(rq);
	unsigned long now;

	now = jiffies;
	blur_log_blurq(blurd, blurq, "complete rqnoidle %d",
		     !!(rq->cmd_flags & REQ_NOIDLE));

	blur_update_hw_tag(blurd);

	WARN_ON(!blurd->rq_in_driver);
	WARN_ON(!blurq->dispatched);
	blurd->rq_in_driver--;
	blurq->dispatched--;
	(RQ_blurG(rq))->dispatched--;
	blur_blkiocg_update_completion_stats(&blurq->blurg->blkg,
			rq_start_time_ns(rq), rq_io_start_time_ns(rq),
			rq_data_dir(rq), rq_is_sync(rq));

	blurd->rq_in_flight[blur_blurq_sync(blurq)]--;

	if (sync) {
		struct blur_rb_root *service_tree;

		RQ_CIC(rq)->ttime.last_end_request = now;

		if (blur_blurq_on_rr(blurq))
			service_tree = blurq->service_tree;
		else
			service_tree = service_tree_for(blurq->blurg,
				blurq_prio(blurq), blurq_type(blurq));
		service_tree->ttime.last_end_request = now;
		if (!time_after(rq->start_time + blurd->blur_fifo_expire[1], now))
			blurd->last_delayed_sync = now;
	}

#ifdef CONFIG_blur_GROUP_IOSCHED
	blurq->blurg->ttime.last_end_request = now;
#endif

	/*
	 * If this is the active queue, check if it needs to be expired,
	 * or if we want to idle in case it has no pending requests.
	 */
	if (blurd->active_queue == blurq) {
		const bool blurq_empty = RB_EMPTY_ROOT(&blurq->sort_list);

		if (blur_blurq_slice_new(blurq)) {
			blur_set_prio_slice(blurd, blurq);
			blur_clear_blurq_slice_new(blurq);
		}

		/*
		 * Should we wait for next request to come in before we expire
		 * the queue.
		 */
		if (blur_should_wait_busy(blurd, blurq)) {
			unsigned long extend_sl = blurd->blur_slice_idle;
			if (!blurd->blur_slice_idle)
				extend_sl = blurd->blur_group_idle;
			blurq->slice_end = jiffies + extend_sl;
			blur_mark_blurq_wait_busy(blurq);
			blur_log_blurq(blurd, blurq, "will busy wait");
		}

		/*
		 * Idling is not enabled on:
		 * - expired queues
		 * - idle-priority queues
		 * - async queues
		 * - queues with still some requests queued
		 * - when there is a close cooperator
		 */
		if (blur_slice_used(blurq) || blur_class_idle(blurq))
			blur_slice_expired(blurd, 1);
		else if (sync && blurq_empty &&
			 !blur_close_cooperator(blurd, blurq)) {
			blur_arm_slice_timer(blurd);
		}
	}

	if (!blurd->rq_in_driver)
		blur_schedule_dispatch(blurd);
}

static inline int __blur_may_queue(struct blur_queue *blurq)
{
	if (blur_blurq_wait_request(blurq) && !blur_blurq_must_alloc_slice(blurq)) {
		blur_mark_blurq_must_alloc_slice(blurq);
		return ELV_MQUEUE_MUST;
	}

	return ELV_MQUEUE_MAY;
}

static int blur_may_queue(struct request_queue *q, int rw)
{
	struct blur_data *blurd = q->elevator->elevator_data;
	struct task_struct *tsk = current;
	struct blur_io_cq *cic;
	struct blur_queue *blurq;

	/*
	 * don't force setup of a queue from here, as a call to may_queue
	 * does not necessarily imply that a request actually will be queued.
	 * so just lookup a possibly existing queue, or return 'may queue'
	 * if that fails
	 */
	cic = blur_cic_lookup(blurd, tsk->io_context);
	if (!cic)
		return ELV_MQUEUE_MAY;

	blurq = cic_to_blurq(cic, rw_is_sync(rw));
	if (blurq) {
		blur_init_prio_data(blurq, cic->icq.ioc);

		return __blur_may_queue(blurq);
	}

	return ELV_MQUEUE_MAY;
}

/*
 * queue lock held here
 */
static void blur_put_request(struct request *rq)
{
	struct blur_queue *blurq = RQ_blurQ(rq);

	if (blurq) {
		const int rw = rq_data_dir(rq);

		BUG_ON(!blurq->allocated[rw]);
		blurq->allocated[rw]--;

		/* Put down rq reference on blurg */
		blur_put_blurg(RQ_blurG(rq));
		rq->elv.priv[0] = NULL;
		rq->elv.priv[1] = NULL;

		blur_put_queue(blurq);
	}
}

static struct blur_queue *
blur_merge_blurqs(struct blur_data *blurd, struct blur_io_cq *cic,
		struct blur_queue *blurq)
{
	blur_log_blurq(blurd, blurq, "merging with queue %p", blurq->new_blurq);
	cic_set_blurq(cic, blurq->new_blurq, 1);
	blur_mark_blurq_coop(blurq->new_blurq);
	blur_put_queue(blurq);
	return cic_to_blurq(cic, 1);
}

/*
 * Returns NULL if a new blurq should be allocated, or the old blurq if this
 * was the last process referring to said blurq.
 */
static struct blur_queue *
split_blurq(struct blur_io_cq *cic, struct blur_queue *blurq)
{
	if (blurq_process_refs(blurq) == 1) {
		blurq->pid = current->pid;
		blur_clear_blurq_coop(blurq);
		blur_clear_blurq_split_coop(blurq);
		return blurq;
	}

	cic_set_blurq(cic, NULL, 1);

	blur_put_cooperator(blurq);

	blur_put_queue(blurq);
	return NULL;
}
/*
 * Allocate blur data structures associated with this request.
 */
static int
blur_set_request(struct request_queue *q, struct request *rq, gfp_t gfp_mask)
{
	struct blur_data *blurd = q->elevator->elevator_data;
	struct blur_io_cq *cic = icq_to_cic(rq->elv.icq);
	const int rw = rq_data_dir(rq);
	const bool is_sync = rq_is_sync(rq);
	struct blur_queue *blurq;
	unsigned int changed;

	might_sleep_if(gfp_mask & __GFP_WAIT);

	spin_lock_irq(q->queue_lock);

	/* handle changed notifications */
	changed = icq_get_changed(&cic->icq);
	if (unlikely(changed & ICQ_IOPRIO_CHANGED))
		changed_ioprio(cic);
#ifdef CONFIG_blur_GROUP_IOSCHED
	if (unlikely(changed & ICQ_CGROUP_CHANGED))
		changed_cgroup(cic);
#endif

new_queue:
	blurq = cic_to_blurq(cic, is_sync);
	if (!blurq || blurq == &blurd->oom_blurq) {
		blurq = blur_get_queue(blurd, is_sync, cic->icq.ioc, gfp_mask);
		cic_set_blurq(cic, blurq, is_sync);
	} else {
		/*
		 * If the queue was seeky for too long, break it apart.
		 */
		if (blur_blurq_coop(blurq) && blur_blurq_split_coop(blurq)) {
			blur_log_blurq(blurd, blurq, "breaking apart blurq");
			blurq = split_blurq(cic, blurq);
			if (!blurq)
				goto new_queue;
		}

		/*
		 * Check to see if this queue is scheduled to merge with
		 * another, closely cooperating queue.  The merging of
		 * queues happens here as it must be done in process context.
		 * The reference on new_blurq was taken in merge_blurqs.
		 */
		if (blurq->new_blurq)
			blurq = blur_merge_blurqs(blurd, cic, blurq);
	}

	blurq->allocated[rw]++;

	blurq->ref++;
	rq->elv.priv[0] = blurq;
	rq->elv.priv[1] = blur_ref_get_blurg(blurq->blurg);
	spin_unlock_irq(q->queue_lock);
	return 0;
}

static void blur_kick_queue(struct work_struct *work)
{
	struct blur_data *blurd =
		container_of(work, struct blur_data, unplug_work);
	struct request_queue *q = blurd->queue;

	spin_lock_irq(q->queue_lock);
	__blk_run_queue(blurd->queue);
	spin_unlock_irq(q->queue_lock);
}

/*
 * Timer running if the active_queue is currently idling inside its time slice
 */
static void blur_idle_slice_timer(unsigned long data)
{
	struct blur_data *blurd = (struct blur_data *) data;
	struct blur_queue *blurq;
	unsigned long flags;
	int timed_out = 1;

	blur_log(blurd, "idle timer fired");

	spin_lock_irqsave(blurd->queue->queue_lock, flags);

	blurq = blurd->active_queue;
	if (blurq) {
		timed_out = 0;

		/*
		 * We saw a request before the queue expired, let it through
		 */
		if (blur_blurq_must_dispatch(blurq))
			goto out_kick;

		/*
		 * expired
		 */
		if (blur_slice_used(blurq))
			goto expire;

		/*
		 * only expire and reinvoke request handler, if there are
		 * other queues with pending requests
		 */
		if (!blurd->busy_queues)
			goto out_cont;

		/*
		 * not expired and it has a request pending, let it dispatch
		 */
		if (!RB_EMPTY_ROOT(&blurq->sort_list))
			goto out_kick;

		/*
		 * Queue depth flag is reset only when the idle didn't succeed
		 */
		blur_clear_blurq_deep(blurq);
	}
expire:
	blur_slice_expired(blurd, timed_out);
out_kick:
	blur_schedule_dispatch(blurd);
out_cont:
	spin_unlock_irqrestore(blurd->queue->queue_lock, flags);
}

static void blur_shutdown_timer_wq(struct blur_data *blurd)
{
	del_timer_sync(&blurd->idle_slice_timer);
	cancel_work_sync(&blurd->unplug_work);
}

static void blur_put_async_queues(struct blur_data *blurd)
{
	int i;

	for (i = 0; i < IOPRIO_BE_NR; i++) {
		if (blurd->async_blurq[0][i])
			blur_put_queue(blurd->async_blurq[0][i]);
		if (blurd->async_blurq[1][i])
			blur_put_queue(blurd->async_blurq[1][i]);
	}

	if (blurd->async_idle_blurq)
		blur_put_queue(blurd->async_idle_blurq);
}

static void blur_exit_queue(struct elevator_queue *e)
{
	struct blur_data *blurd = e->elevator_data;
	struct request_queue *q = blurd->queue;
	bool wait = false;

	blur_shutdown_timer_wq(blurd);

	spin_lock_irq(q->queue_lock);

	if (blurd->active_queue)
		__blur_slice_expired(blurd, blurd->active_queue, 0);

	blur_put_async_queues(blurd);
	blur_release_blur_groups(blurd);

	/*
	 * If there are groups which we could not unlink from blkcg list,
	 * wait for a rcu period for them to be freed.
	 */
	if (blurd->nr_blkcg_linked_grps)
		wait = true;

	spin_unlock_irq(q->queue_lock);

	blur_shutdown_timer_wq(blurd);

	/*
	 * Wait for blurg->blkg->key accessors to exit their grace periods.
	 * Do this wait only if there are other unlinked groups out
	 * there. This can happen if cgroup deletion path claimed the
	 * responsibility of cleaning up a group before queue cleanup code
	 * get to the group.
	 *
	 * Do not call synchronize_rcu() unconditionally as there are drivers
	 * which create/delete request queue hundreds of times during scan/boot
	 * and synchronize_rcu() can take significant time and slow down boot.
	 */
	if (wait)
		synchronize_rcu();

#ifdef CONFIG_blur_GROUP_IOSCHED
	/* Free up per cpu stats for root group */
	free_percpu(blurd->root_group.blkg.stats_cpu);
#endif
	kfree(blurd);
}

static void *blur_init_queue(struct request_queue *q)
{
	struct blur_data *blurd;
	int i, j;
	struct blur_group *blurg;
	struct blur_rb_root *st;

	blurd = kmalloc_node(sizeof(*blurd), GFP_KERNEL | __GFP_ZERO, q->node);
	if (!blurd)
		return NULL;

	/* Init root service tree */
	blurd->grp_service_tree = blur_RB_ROOT;

	/* Init root group */
	blurg = &blurd->root_group;
	for_each_blurg_st(blurg, i, j, st)
		*st = blur_RB_ROOT;
	RB_CLEAR_NODE(&blurg->rb_node);

	/* Give preference to root group over other groups */
	blurg->weight = 2*BLKIO_WEIGHT_DEFAULT;

#ifdef CONFIG_blur_GROUP_IOSCHED
	/*
	 * Set root group reference to 2. One reference will be dropped when
	 * all groups on blurd->blurg_list are being deleted during queue exit.
	 * Other reference will remain there as we don't want to delete this
	 * group as it is statically allocated and gets destroyed when
	 * throtl_data goes away.
	 */
	blurg->ref = 2;

	if (blkio_alloc_blkg_stats(&blurg->blkg)) {
		kfree(blurg);
		kfree(blurd);
		return NULL;
	}

	rcu_read_lock();

	blur_blkiocg_add_blkio_group(&blkio_root_cgroup, &blurg->blkg,
					(void *)blurd, 0);
	rcu_read_unlock();
	blurd->nr_blkcg_linked_grps++;

	/* Add group on blurd->blurg_list */
	hlist_add_head(&blurg->blurd_node, &blurd->blurg_list);
#endif
	/*
	 * Not strictly needed (since RB_ROOT just clears the node and we
	 * zeroed blurd on alloc), but better be safe in case someone decides
	 * to add magic to the rb code
	 */
	for (i = 0; i < blur_PRIO_LISTS; i++)
		blurd->prio_trees[i] = RB_ROOT;

	/*
	 * Our fallback blurq if blur_find_alloc_queue() runs into OOM issues.
	 * Grab a permanent reference to it, so that the normal code flow
	 * will not attempt to free it.
	 */
	blur_init_blurq(blurd, &blurd->oom_blurq, 1, 0);
	blurd->oom_blurq.ref++;
	blur_link_blurq_blurg(&blurd->oom_blurq, &blurd->root_group);

	blurd->queue = q;

	init_timer(&blurd->idle_slice_timer);
	blurd->idle_slice_timer.function = blur_idle_slice_timer;
	blurd->idle_slice_timer.data = (unsigned long) blurd;

	INIT_WORK(&blurd->unplug_work, blur_kick_queue);

	blurd->blur_quantum = blur_quantum;
	blurd->blur_fifo_expire[0] = blur_fifo_expire[0];
	blurd->blur_fifo_expire[1] = blur_fifo_expire[1];
	blurd->blur_back_max = blur_back_max;
	blurd->blur_back_penalty = blur_back_penalty;
	blurd->blur_slice[0] = blur_slice_async;
	blurd->blur_slice[1] = blur_slice_sync;
	blurd->blur_target_latency = blur_target_latency;
	blurd->blur_slice_async_rq = blur_slice_async_rq;
	blurd->blur_slice_idle = blur_slice_idle;
	blurd->blur_group_idle = blur_group_idle;
	blurd->blur_latency = 1;
	blurd->hw_tag = -1;
	/*
	 * we optimistically start assuming sync ops weren't delayed in last
	 * second, in order to have larger depth for async operations.
	 */
	blurd->last_delayed_sync = jiffies - HZ;
	return blurd;
}

/*
 * sysfs parts below -->
 */
static ssize_t
blur_var_show(unsigned int var, char *page)
{
	return sprintf(page, "%d\n", var);
}

static ssize_t
blur_var_store(unsigned int *var, const char *page, size_t count)
{
	char *p = (char *) page;

	*var = simple_strtoul(p, &p, 10);
	return count;
}

#define SHOW_FUNCTION(__FUNC, __VAR, __CONV)				\
static ssize_t __FUNC(struct elevator_queue *e, char *page)		\
{									\
	struct blur_data *blurd = e->elevator_data;			\
	unsigned int __data = __VAR;					\
	if (__CONV)							\
		__data = jiffies_to_msecs(__data);			\
	return blur_var_show(__data, (page));				\
}
SHOW_FUNCTION(blur_quantum_show, blurd->blur_quantum, 0);
SHOW_FUNCTION(blur_fifo_expire_sync_show, blurd->blur_fifo_expire[1], 1);
SHOW_FUNCTION(blur_fifo_expire_async_show, blurd->blur_fifo_expire[0], 1);
SHOW_FUNCTION(blur_back_seek_max_show, blurd->blur_back_max, 0);
SHOW_FUNCTION(blur_back_seek_penalty_show, blurd->blur_back_penalty, 0);
SHOW_FUNCTION(blur_slice_idle_show, blurd->blur_slice_idle, 1);
SHOW_FUNCTION(blur_group_idle_show, blurd->blur_group_idle, 1);
SHOW_FUNCTION(blur_slice_sync_show, blurd->blur_slice[1], 1);
SHOW_FUNCTION(blur_slice_async_show, blurd->blur_slice[0], 1);
SHOW_FUNCTION(blur_slice_async_rq_show, blurd->blur_slice_async_rq, 0);
SHOW_FUNCTION(blur_low_latency_show, blurd->blur_latency, 0);
SHOW_FUNCTION(blur_target_latency_show, blurd->blur_target_latency, 1);
#undef SHOW_FUNCTION

#define STORE_FUNCTION(__FUNC, __PTR, MIN, MAX, __CONV)			\
static ssize_t __FUNC(struct elevator_queue *e, const char *page, size_t count)	\
{									\
	struct blur_data *blurd = e->elevator_data;			\
	unsigned int __data;						\
	int ret = blur_var_store(&__data, (page), count);		\
	if (__data < (MIN))						\
		__data = (MIN);						\
	else if (__data > (MAX))					\
		__data = (MAX);						\
	if (__CONV)							\
		*(__PTR) = msecs_to_jiffies(__data);			\
	else								\
		*(__PTR) = __data;					\
	return ret;							\
}
STORE_FUNCTION(blur_quantum_store, &blurd->blur_quantum, 1, UINT_MAX, 0);
STORE_FUNCTION(blur_fifo_expire_sync_store, &blurd->blur_fifo_expire[1], 1,
		UINT_MAX, 1);
STORE_FUNCTION(blur_fifo_expire_async_store, &blurd->blur_fifo_expire[0], 1,
		UINT_MAX, 1);
STORE_FUNCTION(blur_back_seek_max_store, &blurd->blur_back_max, 0, UINT_MAX, 0);
STORE_FUNCTION(blur_back_seek_penalty_store, &blurd->blur_back_penalty, 1,
		UINT_MAX, 0);
STORE_FUNCTION(blur_slice_idle_store, &blurd->blur_slice_idle, 0, UINT_MAX, 1);
STORE_FUNCTION(blur_group_idle_store, &blurd->blur_group_idle, 0, UINT_MAX, 1);
STORE_FUNCTION(blur_slice_sync_store, &blurd->blur_slice[1], 1, UINT_MAX, 1);
STORE_FUNCTION(blur_slice_async_store, &blurd->blur_slice[0], 1, UINT_MAX, 1);
STORE_FUNCTION(blur_slice_async_rq_store, &blurd->blur_slice_async_rq, 1,
		UINT_MAX, 0);
STORE_FUNCTION(blur_low_latency_store, &blurd->blur_latency, 0, 1, 0);
STORE_FUNCTION(blur_target_latency_store, &blurd->blur_target_latency, 1, UINT_MAX, 1);
#undef STORE_FUNCTION

#define blur_ATTR(name) \
	__ATTR(name, S_IRUGO|S_IWUSR, blur_##name##_show, blur_##name##_store)

static struct elv_fs_entry blur_attrs[] = {
	blur_ATTR(quantum),
	blur_ATTR(fifo_expire_sync),
	blur_ATTR(fifo_expire_async),
	blur_ATTR(back_seek_max),
	blur_ATTR(back_seek_penalty),
	blur_ATTR(slice_sync),
	blur_ATTR(slice_async),
	blur_ATTR(slice_async_rq),
	blur_ATTR(slice_idle),
	blur_ATTR(group_idle),
	blur_ATTR(low_latency),
	blur_ATTR(target_latency),
	__ATTR_NULL
};

static struct elevator_type iosched_blur = {
	.ops = {
		.elevator_merge_fn = 		blur_merge,
		.elevator_merged_fn =		blur_merged_request,
		.elevator_merge_req_fn =	blur_merged_requests,
		.elevator_allow_merge_fn =	blur_allow_merge,
		.elevator_bio_merged_fn =	blur_bio_merged,
		.elevator_dispatch_fn =		blur_dispatch_requests,
		.elevator_add_req_fn =		blur_insert_request,
		.elevator_activate_req_fn =	blur_activate_request,
		.elevator_deactivate_req_fn =	blur_deactivate_request,
		.elevator_completed_req_fn =	blur_completed_request,
		.elevator_former_req_fn =	elv_rb_former_request,
		.elevator_latter_req_fn =	elv_rb_latter_request,
		.elevator_init_icq_fn =		blur_init_icq,
		.elevator_exit_icq_fn =		blur_exit_icq,
		.elevator_set_req_fn =		blur_set_request,
		.elevator_put_req_fn =		blur_put_request,
		.elevator_may_queue_fn =	blur_may_queue,
		.elevator_init_fn =		blur_init_queue,
		.elevator_exit_fn =		blur_exit_queue,
	},
	.icq_size	=	sizeof(struct blur_io_cq),
	.icq_align	=	__alignof__(struct blur_io_cq),
	.elevator_attrs =	blur_attrs,
	.elevator_name	=	"blur",
	.elevator_owner =	THIS_MODULE,
};

#ifdef CONFIG_blur_GROUP_IOSCHED
static struct blkio_policy_type blkio_policy_blur = {
	.ops = {
		.blkio_unlink_group_fn =	blur_unlink_blkio_group,
		.blkio_update_group_weight_fn =	blur_update_blkio_group_weight,
	},
	.plid = BLKIO_POLICY_PROP,
};
#else
static struct blkio_policy_type blkio_policy_blur;
#endif

static int __init blur_init(void)
{
	int ret;

	/*
	 * could be 0 on HZ < 1000 setups
	 */
	if (!blur_slice_async)
		blur_slice_async = 1;
	if (!blur_slice_idle)
		blur_slice_idle = 1;

#ifdef CONFIG_blur_GROUP_IOSCHED
	if (!blur_group_idle)
		blur_group_idle = 1;
#else
		blur_group_idle = 0;
#endif
	blur_pool = KMEM_CACHE(blur_queue, 0);
	if (!blur_pool)
		return -ENOMEM;

	ret = elv_register(&iosched_blur);
	if (ret) {
		kmem_cache_destroy(blur_pool);
		return ret;
	}

	blkio_policy_register(&blkio_policy_blur);

	return 0;
}

static void __exit blur_exit(void)
{
	blkio_policy_unregister(&blkio_policy_blur);
	elv_unregister(&iosched_blur);
	kmem_cache_destroy(blur_pool);
}

module_init(blur_init);
module_exit(blur_exit);

MODULE_AUTHOR("thanhphat<thanhphat1299@gmail.com>");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Completely Fair Queueing IO scheduler");
