#pragma once
// bounded_queue.h
// Thread-safe, capacity-bounded, blocking FIFO queue for the pipeline stages.
//
// Producers call push() – blocks when at capacity.
// Consumers call pop()  – blocks until an item is available.
// close() unblocks all waiters: push() returns false, pop() drains then
// returns false once empty.  This is the end-of-stream signal.

#include <condition_variable>
#include <mutex>
#include <queue>
#include <cstddef>

template<typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t capacity) : cap_(capacity), closed_(false) {}

    // Push an item.  Blocks while full (and not closed).
    // Returns false if the queue has been closed – caller should stop.
    bool push(T item) {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_space_.wait(lk, [this]{ return buf_.size() < cap_ || closed_; });
        if (closed_) return false;
        buf_.push(std::move(item));
        lk.unlock();
        cv_item_.notify_one();
        return true;
    }

    // Pop an item into `out`.  Blocks while empty (and not closed).
    // Returns false when the queue is closed AND empty – no more items.
    bool pop(T& out) {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_item_.wait(lk, [this]{ return !buf_.empty() || closed_; });
        if (buf_.empty()) return false;
        out = std::move(buf_.front());
        buf_.pop();
        lk.unlock();
        cv_space_.notify_one();
        return true;
    }

    // Signal that no more items will be pushed.
    // Unblocks any threads waiting in push() or pop().
    void close() {
        {
            std::unique_lock<std::mutex> lk(mtx_);
            closed_ = true;
        }
        cv_item_.notify_all();
        cv_space_.notify_all();
    }

    bool is_closed() const {
        std::unique_lock<std::mutex> lk(mtx_);
        return closed_;
    }

    // Current item count -- for queue-depth instrumentation. A snapshot,
    // not synchronized with any particular push/pop (the queue is
    // concurrently mutated by other threads), but adequate for periodic
    // depth sampling/reporting.
    size_t size() const {
        std::unique_lock<std::mutex> lk(mtx_);
        return buf_.size();
    }

private:
    std::queue<T>           buf_;
    const size_t            cap_;
    mutable std::mutex      mtx_;
    std::condition_variable cv_item_;   // signalled when an item is added
    std::condition_variable cv_space_;  // signalled when space opens up
    bool                    closed_;
};
