#include "LocalMeanSqrFilter.h"

#define MY_LOGD(fmt, ...)    do { printf("[%s] "fmt, __FUNCTION__, __VA_ARGS__); printf("\n"); } while (0)
#define MY_LOGW              MY_LOGD
#define MY_LOGE              MY_LOGD

using namespace cv;

class LMVFilterImpl
{
public:
    virtual                     ~LMVFilterImpl() {}
    void                        filter( Mat *pmDst, int r, int level );

protected:
    int                         Idepth;
    bool                        initWithData;
    
private:
    //virtual void                cptSppANDSqrSqq( bool padded ) = 0; 
    //virtual void                cptLocalSumANDSqrSum( int rowIdx, int r ) = 0;
    virtual void                filterSingleChannel( Mat *pmDst, int r, int level) = 0;
};

class LMVFilterMono : public LMVFilterImpl
{
public:
                                LMVFilterMono( Mat &mSrc );
	virtual                    ~LMVFilterMono();

private:
    virtual void                filterSingleChannel( Mat *pmDst, int r, int level );
    virtual void                cptSppANDSqrSqq( bool padded );
	virtual void                cptLocalSumANDSqrSum( int rowIdx, int r );
    void                        test(void);

private:
    int                         *pMyMem;
    int                         *pPreLSum;     //previous local sum
    int                         *pPreLSqrSum;  //previous local square sum
    int                         *pCurLSum;     //current local sum
    int                         *pCurLSqrSum;  //current local square sum

    int                         nRows;
    int                         nCols;
    Mat                         mSrc;
    Mat                         mSpp;          //superposition sum of each row
    Mat                         mSqrSpp;       //superposition sum of square of each row
};

/* overrigde data in mSrc */
void LMVFilterImpl::filter( Mat *pmDst, int r, int level )
{
    filterSingleChannel( pmDst, r, level );
}

LMVFilterMono::LMVFilterMono(Mat &mSrc) : mSrc(mSrc)
{
    nCols = mSrc.cols;
    nRows = mSrc.rows;

    pMyMem = new int[nCols*4];
    memset(pMyMem, 0, nCols*4*sizeof(int));

    pPreLSum    = pMyMem + nCols*0;  //previous local sum
    pPreLSqrSum = pMyMem + nCols*1;  //previous local square sum
    pCurLSum    = pMyMem + nCols*2;  //current local sum
    pCurLSqrSum = pMyMem + nCols*3;  //current local square sum

    mSpp.create( mSrc.size(), CV_32SC1 );
    mSqrSpp.create( mSrc.size(), CV_32SC1 );

    initWithData = false;
    //cptSppANDSqrSqq( 0 );
}

LMVFilterMono::~LMVFilterMono(void)
{
    delete pMyMem;
}


/* 
  Compute superposed results of each pixel and its square vaule in each row.
 
    Src:    12, 23, 32, 11, 223, 32,  231, 34,  ......
  Superposed result:
    Padded:  0, 12, 35, 67, 78,  301, 333, 564, 598, ......
    no pad: 12, 35, 67, 78, 301, 333, 564, 598, ......
*/
void LMVFilterMono::cptSppANDSqrSqq( bool padded )
{
    int nRows = mSrc.rows;
    int nCols = (padded == 1) ? mSrc.cols : mSrc.cols+1;

    if ( (mSpp.cols != nCols) || (mSpp.rows != nRows) )
        mSpp.create( nRows, nCols, CV_32SC1 );

    if ( (mSqrSpp.cols != nCols) || (mSqrSpp.rows != nRows) )
        mSqrSpp.create( nRows, nCols, CV_32SC1 );

    if ( padded == 1 )
    {
        for (int i=0; i<nRows; i++)
        {  
            uchar* pSrc = mSrc.ptr<uchar>(i);
            int *pSppRow    = mSpp.ptr<int>(i);
            int *pSppSqrRow = mSqrSpp.ptr<int>(i);

            for( int j=1; j<nCols; j++)
            {
                pSppRow[j]    = pSppRow[j-1]    + pSrc[j-1];
                pSppSqrRow[j] = pSppSqrRow[j-1] + pSrc[j-1]*pSrc[j-1];
            }
        }
    }else
    {
        for (int i=0; i<nRows; i++)
        {  
            uchar* pSrc = mSrc.ptr<uchar>(i);
            int *pSppRow    = mSpp.ptr<int>(i);
            int *pSppSqrRow = mSqrSpp.ptr<int>(i);

            pSppRow[0]    = pSrc[0];
            pSppSqrRow[0] = pSrc[0]*pSrc[0];

            for( int j=1; j<nCols; j++)
            {
                pSppRow[j]    = pSppRow[j-1]    + pSrc[j];
                pSppSqrRow[j] = pSppSqrRow[j-1] + pSrc[j]*pSrc[j];
                //pSppSqrRow[j] = pSppSqrRow[j-1] + table_u8Squares[pSrc[j]];
            }
        }
    }
}


/* 
'rowIdx' should not smaller than 'r' 
 
e.g. cptLocalSumANDSqrSum(5,4) 
     '-' are origin values, '*' are computed values, '~' need to be used
    0: ----------------------------------
    1: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    2: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    3: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    4: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    5: ~~~~**************************~~~~
    6: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    7: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    8: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    9: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   10: ----------------------------------
*/
void LMVFilterMono::cptLocalSumANDSqrSum( int rowIdx, int r )
{
    int nRows = mSrc.rows;
    int nCols = mSrc.cols;
    int kh = r*2 + 1;  //kernel height
    int kw = r*2 + 1;  //kernel width

    int firstRow = rowIdx - r;
    if ( firstRow < 0 )
        firstRow = 0;
    else if ( (firstRow+r) >= nCols )
        firstRow = nRows - rowIdx - 1;

    /* radius+1 section to radius+1 last section, only compute offset value */
    /* ----*******************************---- */
    for( int i=firstRow; i<firstRow+kh; i++ )
    {
        int *pSppRow    = mSpp.ptr<int>(i);
        int *pSqrSppRow = mSqrSpp.ptr<int>(i);

        /* compute r col independently. As spp is not padded, spp[0] is not zero. */
        pCurLSum[r]    += pSppRow[kw-1];
        pCurLSqrSum[r] += pSqrSppRow[kw-1];

        for( int j=r+1; j<nCols-r; j++)
        {
            pCurLSum[j]    += pSppRow[j+r]    - pSppRow[j-r-1];
            pCurLSqrSum[j] += pSqrSppRow[j+r] - pSqrSppRow[j-r-1];
        }
    }

#define COPY 0
#if 0
    /* Copy to 'first col to radius-1 col' */
    for( int j=0; j<r; j++)
    {
        pCurLSum[j]    = pCurLSum[r];
        pCurLSqrSum[j] = pCurLSqrSum[r];
    }
    /* Copy to last radius cols */
    for( int j=nCols-r; j<nCols; j++)
    {
        pCurLSum[j]    = pCurLSum[nCols-r-1];
        pCurLSqrSum[j] = pCurLSqrSum[nCols-r-1];
    }
#endif
#undef COPY
}

void LMVFilterMono::filterSingleChannel( Mat *pmDst, int r, int level )
{
    Mat mDst;
    if ( pmDst == NULL )
        mDst = mSrc;
    else
        mDst = *pmDst;

    if ( initWithData == false )
    {
        cptSppANDSqrSqq( 0 );
        cptLocalSumANDSqrSum( r, r );
    }
    int kh = r*2 + 1;  //kernel height
    int kw = r*2 + 1;  //kernel width
    int r1 = r+1;
    int area = kh*kw;
    int alpha = level * level * 5 + 10;

    /* Start: compute result in r row */
    uchar* pSrc = mSrc.ptr<uchar>(r);
    uchar* pDst = mDst.ptr<uchar>(r);

    for( int j=r; j<nCols-r; j++)
    {
        int mean    = pCurLSum[j] / area;
        int sqrmean = pCurLSqrSum[j] / area;
        //TODO
        pDst[j] = mean;
    }
    /* End: compute result in r row  */

    /* Start: compute result from r+1 row to last r+1 row */
    for (int i=r1; i<nRows-r; i++)
    {  
        pSrc = mSrc.ptr<uchar>(i);
        pDst = mDst.ptr<uchar>(i);

        int *pSppRow_rm     = mSpp.ptr<int>(i-r1);    // the row to be removed
        int *pSppSqrRow_rm  = mSqrSpp.ptr<int>(i-r1); // the row to be removed
        int *pSppRow_add    = mSpp.ptr<int>(i+r);     // the row to be added
        int *pSppSqrRow_add = mSqrSpp.ptr<int>(i+r);  // the row to be added

        /* exchange buf */
        int *temp = pPreLSum; pPreLSum = pCurLSum; pCurLSum = temp;
        temp = pPreLSqrSum; pPreLSqrSum = pCurLSqrSum; pCurLSqrSum = temp;

        pCurLSum[r]    = pPreLSum[r]    + pSppRow_add[r*2]    - pSppRow_rm[r*2];
        pCurLSqrSum[r] = pPreLSqrSum[r] + pSppSqrRow_add[r*2] - pSppSqrRow_add[r*2];

        int mean    = pCurLSum[r] / area;
        int sqrmean = pCurLSqrSum[r] / area;

        //TODO
        pDst[r] = mean;

        for( int j=r1; j<nCols-r; j++)
        {
            int add    = pSppRow_add[j+r] - pSppRow_add[j-r1];
            int sqradd = pSppSqrRow_add[j+r] - pSppSqrRow_add[j-r1];
            int rm     = pSppRow_rm[j+r] - pSppRow_rm[j-r1];
            int sqrrm  = pSppSqrRow_rm[j+r] - pSppSqrRow_rm[j-r1];

            pCurLSum[j]    = pPreLSum[j] + add - rm;
            pCurLSqrSum[j] = pPreLSqrSum[j] + sqradd - sqrrm;

            mean    = pCurLSum[j] / area;
            //sqrmean = pCurLSqrSum[j] / area;

            int lSdv = (pCurLSqrSum[j] - (mean * pCurLSum[j])) / area;

            //TODO
            pDst[j] = (uchar)(alpha*mean/(lSdv+alpha) + lSdv*pSrc[j]/(lSdv+alpha));
            //pDst[j] = lSdv;
        }
    }
    /* End: compute result from r+1 row to last r+1 row */

    //test();
}

void LMVFilterMono::test(void)
{
    MY_LOGD("fopen");
    FILE *file1 = fopen("d:/mean.txt", "w+");
    if( NULL == file1 )
    {
        MY_LOGD("fopen failed");
        return;
    }

    for( int i=0; i<mSrc.rows; i++ )
    {  
        uchar* pSrc = mSrc.ptr<uchar>(i);

        for( int j=0; j<mSrc.cols; j++)
        {
            char buffer[20];
            sprintf(buffer, "%d ", pSrc[j]);
            fwrite(buffer, 1, strlen(buffer), file1);
        }
        fwrite("\n", 1, 1, file1);
    }
    fclose(file1);
}

LMVFilter::LMVFilter( Mat &mSrc )
{
    impl_ = new LMVFilterMono( mSrc );
}

LMVFilter::~LMVFilter()
{
    delete impl_;
}

void LMVFilter::filter( Mat *pmDst, int r, int level )
{
    impl_->filter( pmDst, r, level );
}
