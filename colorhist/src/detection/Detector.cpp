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
      static void declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
      {
        std::cout<<"Detector: declare_io"<<std::endl;
        /* Model = Database model | Object = recognized object */
        inputs.declare(&Detector::model_colorValues_, "model_colorValues", "Color values of the database models.");
        inputs.declare(&Detector::object_colorValues_, "object_colorValues", "Color values of the recognized objects");

        outputs.declare(&Detector::pose_results_, "pose_results", "The results of object recognition");
      }

      int process(const tendrils& inputs, const tendrils& outputs)
      {
        std::cout<<"Detector: process"<<std::endl;
        PoseResult pose_result;
        pose_results_->clear();
        std::vector<cv::Mat>  object_colorValues;
        std::vector<cv::Mat>  model_colorValues;
        object_colorValues = *object_colorValues_;
        model_colorValues = *model_colorValues_;

        std::cout<<"object_colorValues: "<<object_colorValues.size()<<std::endl;
        std::cout<<"model_colorValues: "<<model_colorValues.size()<<std::endl;

        if(!object_colorValues.empty() && !model_colorValues.empty())
        {
          float model_colorhist_array[26][3];
          std::fill(model_colorhist_array[0], model_colorhist_array[0] + 26 * 3, 0);
          float object_colorhist_array[26][3];
          std::fill(object_colorhist_array[0], object_colorhist_array[0] + 26 * 3, 0);
          double result = 0;
          cv::MatND object_colorhist;
          cv::MatND model_colorhist;

          //~ for(unsigned int i=0; i<object_colorValues.size(); i++)
          //~ {
            //~ for(int j=0; j<26; j++)
            //~ {
              //~ std::cout<<"R: "<<object_colorValues[i].at<float>(j,0)<<" G: "<<object_colorValues[i].at<float>(j,1)<<" B: "<<object_colorValues[i].at<float>(j,2)<<std::endl;
            //~ }
          //~ }
          std::cout<<"....................................................."<<std::endl;
          
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
          }
          model_colorhist = cv::MatND(26,3, CV_32F, model_colorhist_array);

		  /* Oben wird keine K-Ebene gebraucht, da es immer nur eine 27|3 Matrix kommt.
		   * Hier brauchen wir die K-Ebene, da es k* 27| Matrizen gibt. 
		   * Diese Werte müssen dann dem entsprechend auch noch umgerechnet werden
		  */
          for(unsigned int i=0; i<object_colorValues.size(); i++)
          {
            for(int j= 0; j<26; j++)
            {
              for (int k = 0; k < 3; k++)
              {
                object_colorhist_array[j][k] += object_colorValues[i].at<float>(j,k);
              }
            }
          }
		  
		  int doc_length = 0;
		  doc_length = object_colorValues.size()*26*3;
		  
		  for(int j= 0; j<26; j++)
          {
            for (int k = 0; k < 3; k++)
            {
              object_colorhist_array[j][k] = (object_colorhist_array[j][k]*100)/doc_length;
            }
          }
          object_colorhist = cv::MatND(26,3, CV_32F, object_colorhist_array);

          //~ for(unsigned int i=0; i<26; i++)
          //~ {
			//~ std::cout<<"R: "<<object_colorhist_array[i][0]<<" G: "<<object_colorhist_array[i][1]<<" B: "<<object_colorhist_array[i][2]<<std::endl;
          //~ }

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
        std::cout<<"________________________________________"<<std::endl;
        return ecto::OK;
      }

    private:
      ecto::spore<std::vector<object_recognition_core::common::PoseResult> > pose_results_;
      ecto::spore<std::vector<cv::Mat> > object_colorValues_;
      ecto::spore<std::vector<cv::Mat> > model_colorValues_;
  };
}

ECTO_CELL(colorhist_detection, colorhist::Detector, "Detector", "Reads a colorhist model from the db")
