#pragma once
// thread_pool.h  –  header-only persistent thread pool (C++14)
//
// Worker threads are created once and live until the pool is destroyed.
// submit() wraps any callable in a shared_ptr<packaged_task> so it can be
// stored in a std::function<void()> (packaged_task itself is not copyable).

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

class ThreadPool {
public:
    explicit ThreadPool(int n) : stop_(false) {
        for (int i = 0; i < n; i++)
            workers_.emplace_back([this]{ worker_loop(); });
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lk(mtx_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }

    // Enqueue a callable with no arguments.  Returns std::future<R> where
    // R = return type of f().  Works in C++14.
    template<class F>
    auto submit(F&& f) -> std::future<decltype(f())> {
        using R = decltype(f());
        // shared_ptr makes the task copyable so std::function can hold it
        auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
        std::future<R> fut = task->get_future();
        {
            std::unique_lock<std::mutex> lk(mtx_);
            if (stop_)
                throw std::runtime_error("ThreadPool: submit after shutdown");
            queue_.push([task]{ (*task)(); });
        }
        cv_.notify_one();
        return fut;
    }

    int size() const { return static_cast<int>(workers_.size()); }

private:
    void worker_loop() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait(lk, [this]{ return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty()) return;
                task = std::move(queue_.front());
                queue_.pop();
            }
            task();
        }
    }

    std::vector<std::thread>           workers_;
    std::queue<std::function<void()>>  queue_;
    std::mutex                         mtx_;
    std::condition_variable            cv_;
    bool                               stop_;
};
