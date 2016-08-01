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
		/* Model = Database model | Object = recognized object */
        inputs.declare(&Detector::model_colorValues_, "model_colorValues", "Color values of the database models.");
        inputs.declare(&Detector::object_colorValues_, "object_colorValues", "Color values of the recognized objects");

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
        std::vector<cv::Mat>  object_colorValues;
        std::vector<cv::Mat>  model_colorValues;
        object_colorValues = *object_colorValues_;
        model_colorValues = *model_colorValues_;
        
        if(object_colorValues.empty())
			std::cout<<object_colorValues.size()<<std::endl;
        
		if(!object_colorValues.empty() && !model_colorValues.empty())
		{
			float model_colorhist_array[26][3];
			float object_colorhist_array[26][3];
			double result = 0;
			cv::MatND object_colorhist;
			cv::MatND model_colorhist;
			
			/*
			 * colorValues.size() = 27 but the last row contains the avg. colors
			 * and at the moment we dont need the avg. colors
			*/
			for(unsigned int i=0; i<model_colorValues.size()-1; i++)
			{
			  for (int j = 0; j < 3; ++j)
			  {
				model_colorhist_array[i][j] = model_colorValues[i].at<float>(0,j);
			  }		  
			  model_colorhist = cv::MatND(26,3, CV_32F, model_colorhist_array);
			}   
			
			for(unsigned int i=0; i<object_colorValues.size(); i++)
			{
			  for (int j = 0; j < 3; ++j)
			  {
				object_colorhist_array[i][j] = object_colorValues[i].at<float>(0,j);
			  }		  
			  object_colorhist = cv::MatND(26,3, CV_32F, object_colorhist_array);
			}
			
			/* Hier muss später noch ein model_colorhist durch object_colorhist ausgetauscht werden */
			result = methodCorrelation(model_colorhist, model_colorhist);
			std::cout<<"1: "<<result<<std::endl;

			result = methodChiSquare(model_colorhist, model_colorhist);
			std::cout<<"2: "<<result<<std::endl;

			result = methodIntersection(model_colorhist, model_colorhist);
			std::cout<<"3: "<<result<<std::endl;

			result = methodBhattacharyya(model_colorhist, model_colorhist);
			std::cout<<"4: "<<result<<std::endl;
		}
		
        pose_results_->push_back(pose_result);

        return ecto::OK;
      }

    private:
      object_recognition_core::db::ObjectDbPtr db_;
      ecto::spore<object_recognition_core::db::ObjectDbParameters> db_params_;
      ecto::spore<std::vector<object_recognition_core::common::PoseResult> > pose_results_;
      ecto::spore<std::vector<cv::Mat> > object_colorValues_;
	  ecto::spore<std::vector<cv::Mat> > model_colorValues_;
  };
}

ECTO_CELL(colorhist_detection, colorhist::Detector, "Detector", "Reads a colorhist model from the db")
