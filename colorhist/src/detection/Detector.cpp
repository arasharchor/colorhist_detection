#include <string>
#include <fstream>
#include <iostream>

#include <boost/foreach.hpp>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ecto/ecto.hpp>

#include <object_recognition_core/common/types.h>
#include <object_recognition_core/db/db.h>
#include <object_recognition_core/common/pose_result.h>
#include <object_recognition_core/db/opencv.h>

#include "db_colorhist.h"
#include "detection.h"

using ecto::tendrils;
using object_recognition_core::common::PoseResult;

namespace colorhist
{
  /* Cell that loads a colorhist model from the DB */
  struct Detector
  {
  public:
    static void declare_params(tendrils& params)
    {
      params.declare(&Detector::db_params_, "db_params", "The DB parameters").required(true);
    }
    
	static void declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
    {  
      inputs.declare(&Detector::colorValues_, "colorValues", "Color values in a matrix.");
      
      outputs.declare(&Detector::pose_results_, "pose_results", "The results of object recognition");
    }
    
    void configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
    {
      db_params_ = params["db_params"];
      db_ = db_params_->generateDb();
    }

    int process(const tendrils& inputs, const tendrils& outputs)
    {
	  PoseResult pose_result;
      pose_results_->clear();    
      std::vector<cv::Mat>  colorValues;
      colorValues = *colorValues_;
      
      float colorhist_array[26][3];
	  double result = 0;
	  cv::MatND M;
	  
	  /* 
	   * colorValues.size() = 27 but the last row contains the avg. colors 
	   * and at the moment we dont need the avg. colors 
	  */	 
      for(unsigned int i=0; i<colorValues.size()-1; i++)
	  {
		  for (int j = 0; j < 3; ++j)
		  {
			   colorhist_array[i][j] = colorValues[i].at<float>(0,j);
		  }
		  M= cv::MatND(26,3, CV_32F, colorhist_array);
      }
      
      result = methodCorrelation(M, M);
      std::cout<<"1: "<<result<<std::endl;
      
      result = methodChiSquare(M, M);
      std::cout<<"2: "<<result<<std::endl;
      
      result = methodIntersection(M, M);
      std::cout<<"3: "<<result<<std::endl;
      
      result = methodBhattacharyya(M, M);
      std::cout<<"4: "<<result<<std::endl;
      
      pose_results_->push_back(pose_result);
     
      return ecto::OK;
    }
    
  private:
    object_recognition_core::db::ObjectDbPtr db_;
    ecto::spore<std::vector<cv::Mat> > colorValues_;
    ecto::spore<object_recognition_core::db::ObjectDbParameters> db_params_;
    ecto::spore<std::vector<object_recognition_core::common::PoseResult> > pose_results_;
  };
}

ECTO_CELL(colorhist_detection, colorhist::Detector, "Detector", "Reads a colorhist model from the db")  
