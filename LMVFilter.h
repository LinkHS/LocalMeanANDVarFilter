#pragma once

#include "opencv.hpp"

/* Local mean and mean square filter */
class LMVFilterImpl;

class LMVFilter
{
public:
    LMVFilter(cv::Mat &mSrc);
    LMVFilter(int h, int w, int r);
    ~LMVFilter();

    void filter( cv::Mat *pmDst, int r, int level);

private:
    LMVFilterImpl *impl_;
};

