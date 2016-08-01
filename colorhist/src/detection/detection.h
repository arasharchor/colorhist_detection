#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>


bool methodCRG (float frameHist[][3], float modelHist[][3]);

double methodChiQuadratic (cv::MatND frameHist, cv::MatND modelHist);

double methodCorrelation (cv::MatND frameHist, cv::MatND modelHist);

double methodChiSquare (cv::MatND frameHist, cv::MatND modelHist);

double methodIntersection (cv::MatND frameHist, cv::MatND modelHist);

double methodBhattacharyya (cv::MatND frameHist, cv::MatND modelHist);

