#include "detection.h"

using namespace std;
using namespace cv;

/* Wenn noch Zeit ist, könnten noch mehr Algorithmen getestet/implementiert werden */
bool methodCRG (float frameHist[][3], float modelHist[][3])
{
	return true;
}

double methodChiQuadratic (cv::MatND frameHist, cv::MatND modelHist)
{
    return cv::compareHist( frameHist, modelHist, 1);
}
/* ------------------------------------------------------------------------------- */

double methodCorrelation (cv::MatND frameHist, cv::MatND modelHist)
{
	return cv::compareHist( frameHist, modelHist, 0);
}

double methodChiSquare (cv::MatND frameHist, cv::MatND modelHist)
{
    return cv::compareHist( frameHist, modelHist, 1);
}

double methodIntersection (cv::MatND frameHist, cv::MatND modelHist)
{
    return cv::compareHist( frameHist, modelHist, 2);
}

double methodBhattacharyya (cv::MatND frameHist, cv::MatND modelHist)
{
    return cv::compareHist( frameHist, modelHist, 3);
}
