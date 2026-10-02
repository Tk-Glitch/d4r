#pragma once
#include <cstdint>
#include <deque>

// A timeline signal releases every lower value too. Prep can cancel a later
// frame while the CUDA worker is still producing an earlier one, so only the
// completed prefix of admitted frames may be signalled. The caller serializes
// admission and retirement with the same mutex used for the timeline signal.
class D4rFrameCompletion
{
public:
    void admit(uint32_t frame) { pending_.push_back({frame, false}); }

    uint64_t retire(uint32_t frame)
    {
        for (auto& pending : pending_)
            if (pending.frame == frame)
            {
                pending.done = true;
                break;
            }
        while (!pending_.empty() && pending_.front().done)
        {
            completedThrough_ = pending_.front().frame;
            pending_.pop_front();
        }
        return completedThrough_;
    }

private:
    struct Pending { uint32_t frame; bool done; };
    std::deque<Pending> pending_;
    uint64_t completedThrough_ = 0;
};
