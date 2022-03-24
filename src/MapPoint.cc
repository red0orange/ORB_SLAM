/**
* This file is part of ORB-SLAM.
*
* Copyright (C) 2014 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <http://webdiis.unizar.es/~raulmur/orbslam/>
*
* ORB-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include "MapPoint.h"
#include "ORBmatcher.h"
#include "ros/ros.h"

namespace ORB_SLAM
{

long unsigned int MapPoint::nNextId=0;


MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnTrackReferenceForFrame(0), mnLastFrameSeen(0), mnBALocalForKF(0),
    mnLoopPointForKF(0), mnCorrectedByKF(0),mnCorrectedReference(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1),
    mbBad(false), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    mnId=nNextId++;
    mNormalVector = cv::Mat::zeros(3,1,CV_32F);
}

void MapPoint::SetWorldPos(const cv::Mat &Pos)
{
    boost::mutex::scoped_lock lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}

cv::Mat MapPoint::GetWorldPos()
{
    boost::mutex::scoped_lock lock(mMutexPos);
    return mWorldPos.clone();
}

cv::Mat MapPoint::GetNormal()
{
    boost::mutex::scoped_lock lock(mMutexPos);
    return mNormalVector.clone();
}

KeyFrame* MapPoint::GetReferenceKeyFrame()
{
     boost::mutex::scoped_lock lock(mMutexFeatures);
     return mpRefKF;
}

void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    mObservations[pKF]=idx;
}

void MapPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad=false;
    {
        boost::mutex::scoped_lock lock(mMutexFeatures);
        if(mObservations.count(pKF))
        {
            mObservations.erase(pKF);

            if(mpRefKF==pKF)
                mpRefKF=mObservations.begin()->first;

            // If only 2 observations or less, discard point
            if(mObservations.size()<=2)
                bBad=true;
        }
    }

    if(bBad)
        SetBadFlag();
}

map<KeyFrame*, size_t> MapPoint::GetObservations()
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    return mObservations;
}

int MapPoint::Observations()
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    return mObservations.size();
}

void MapPoint::SetBadFlag()
{
    map<KeyFrame*,size_t> obs;
    {
        boost::mutex::scoped_lock lock1(mMutexFeatures);
        boost::mutex::scoped_lock lock2(mMutexPos);
        mbBad=true;
        obs = mObservations;
        mObservations.clear();
    }
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        pKF->EraseMapPointMatch(mit->second);
    }

    mpMap->EraseMapPoint(this);
}

void MapPoint::Replace(MapPoint* pMP)
{
    if(pMP->mnId==this->mnId)
        return;

    map<KeyFrame*,size_t> obs;
    {
        boost::mutex::scoped_lock lock1(mMutexFeatures);
        boost::mutex::scoped_lock lock2(mMutexPos);
        obs=mObservations;
        mObservations.clear();
        mbBad=true;
    }

    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        // Replace measurement in keyframe
        KeyFrame* pKF = mit->first;

        if(!pMP->IsInKeyFrame(pKF))
        {
            pKF->ReplaceMapPointMatch(mit->second, pMP);
            pMP->AddObservation(pKF,mit->second);
        }
        else
        {
            pKF->EraseMapPointMatch(mit->second);
        }
    }

    pMP->ComputeDistinctiveDescriptors();

    mpMap->EraseMapPoint(this);

}

bool MapPoint::isBad()
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    boost::mutex::scoped_lock lock2(mMutexPos);
    return mbBad;
}

void MapPoint::IncreaseVisible()
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    mnVisible++;
}

void MapPoint::IncreaseFound()
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    mnFound++;
}

float MapPoint::GetFoundRatio()
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    return static_cast<float>(mnFound)/mnVisible;
}

void MapPoint::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    // 每个描述子的类型是cv::Mat
    vector<cv::Mat> vDescriptors;

    map<KeyFrame*,size_t> observations;

    {
        boost::mutex::scoped_lock lock1(mMutexFeatures);
        if(mbBad)
            return;
        observations=mObservations;
    }

    if(observations.empty())
        return;

    vDescriptors.reserve(observations.size()); // 设置和observations相同的大小

    // 把每个KeyFrame中该MapPoint对应的描述子取出来放进vDescriptors临时变量中
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        if(!pKF->isBad())
            vDescriptors.push_back(pKF->GetDescriptor(mit->second));
    }

    if(vDescriptors.empty())
        return;

    // 计算vDescriptors中所有描述子的两两距离，距离函数是ORBmatcher::DescriptorDistance
    // Compute distances between them
    const size_t N = vDescriptors.size();

    float Distances[N][N];  // 方矩阵
    for(size_t i=0;i<N;i++)
    {
        Distances[i][i]=0;
        for(size_t j=i+1;j<N;j++)
        {
            int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for(size_t i=0;i<N;i++)  // 遍历每一行
    {
        // 取出每一行到一个vector中并sort
        vector<int> vDists(Distances[i],Distances[i]+N);
        sort(vDists.begin(),vDists.end());
        int median = vDists[0.5*(N-1)];  // 取出中值

        if(median<BestMedian)  // 得到最小的中值，相当于找到中值最小的一行
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    // @note MapPoint理解关键：作者应该想利用 中值小 -> 整体小 的对应，然后得到“对于当前MapPoint，和所有描述子距离最小的最佳描述子”
    {
        boost::mutex::scoped_lock lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();       
    }
}

cv::Mat MapPoint::GetDescriptor()
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    return mDescriptor.clone();
}

int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}

bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
    boost::mutex::scoped_lock lock(mMutexFeatures);
    return (mObservations.count(pKF));
}

void MapPoint::UpdateNormalAndDepth()
{
    map<KeyFrame*,size_t> observations;
    KeyFrame* pRefKF;
    cv::Mat Pos;
    {
        boost::mutex::scoped_lock lock1(mMutexFeatures);
        boost::mutex::scoped_lock lock2(mMutexPos);
        if(mbBad)
            return;
        observations=mObservations;
        pRefKF=mpRefKF;
        Pos = mWorldPos.clone();
    }

    cv::Mat normal = cv::Mat::zeros(3,1,CV_32F);
    int n=0;
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        cv::Mat Owi = pKF->GetCameraCenter();
        cv::Mat normali = mWorldPos - Owi;  // camera 指向该 MapPoint的方向向量
        normal = normal + normali/cv::norm(normali);  // 归一化方向向量，并叠加到normal先
        n++;
    } 

    cv::Mat PC = Pos - pRefKF->GetCameraCenter();
    const float dist = cv::norm(PC); // camera 指向该 MapPoint的方向向量的长度，即camera 到该 MapPoint的距离
    const int level = pRefKF->GetKeyPointScaleLevel(observations[pRefKF]);  // 获得该MapPoint在KeyFrame中对应ORB特征的尺度
    const float scaleFactor = pRefKF->GetScaleFactor();
    const float levelScaleFactor =  pRefKF->GetScaleFactor(level);
    const int nLevels = pRefKF->GetScaleLevels();

    {
        boost::mutex::scoped_lock lock3(mMutexPos);
        mfMinDistance = (1.0f/scaleFactor)*dist / levelScaleFactor;  // 不知道有什么用，经过缩放因子后的dist
        mfMaxDistance = scaleFactor*dist * pRefKF->GetScaleFactor(nLevels-1-level); // 不知道有什么用，经过缩放因子后的dist
        mNormalVector = normal/n;  // 除以总observation数量，得到平均归一化方向向量
    }
}

float MapPoint::GetMinDistanceInvariance()
{
    boost::mutex::scoped_lock lock(mMutexPos);
    return mfMinDistance;
}

float MapPoint::GetMaxDistanceInvariance()
{
    boost::mutex::scoped_lock lock(mMutexPos);
    return mfMaxDistance;
}

} //namespace ORB_SLAM
