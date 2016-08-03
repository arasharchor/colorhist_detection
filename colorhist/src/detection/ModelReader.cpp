#include <string>
#include <vector>
#include <boost/array.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>

#include <boost/foreach.hpp>
#include <ecto/ecto.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/rgbd/rgbd.hpp>
#include <opencv2/opencv.hpp>

#include <object_recognition_core/common/types_eigen.h>
#include <object_recognition_core/common/json.hpp>
#include <object_recognition_core/db/db.h>
#include <object_recognition_core/db/document.h>
#include <object_recognition_core/db/model_utils.h>
#include <object_recognition_core/db/view.h>
#include <object_recognition_core/db/opencv.h>

#include <object_recognition_renderer/renderer3d.h>
#include <object_recognition_renderer/utils.h>

#include "db_colorhist.h"

using namespace cv;
using namespace std;

/* cell storing the 3d points and descriptors while a model is being computed */
struct ModelReader
{
  public:
    static void declare_io(const ecto::tendrils& params, ecto::tendrils& inputs, ecto::tendrils& outputs)
    {
      inputs.declare(&ModelReader::json_db_, "json_db", "The parameters of the DB as a JSON string.").required(true);
      //~ inputs.declare(&ModelReader::object_id_, "object_id", "The id of the object in the DB.").required(true);

      outputs.declare < std::vector<cv::Mat> > ("model_colorValues", "Color values in a matrix");
    }

    int process(const ecto::tendrils& inputs, const ecto::tendrils& outputs)
    {
      /* Get the DB */
      object_recognition_core::db::ObjectDbPtr db =
        object_recognition_core::db::ObjectDbParameters(*json_db_).generateDb();
      object_recognition_core::db::Documents documents =
        object_recognition_core::db::ModelDocuments(db,
            std::vector<object_recognition_core::db::ObjectId>(1, "544ac4d13f955e6ebb22518b3f002734" ), "ColorHist");

      if (documents.empty())
      {
        std::cerr << "Skipping object id"<<std::endl;
        //~ std::cerr << "Skipping object id \"" << *object_id_ << "\" : no mesh in the DB" << std::endl;
        return ecto::OK;
      }

      /* Get the list of _attachments and figure out the original one */
      object_recognition_core::db::Document document = documents[0];
      std::vector<std::string> attachments_names = document.attachment_names();
      std::string possible_name = "colorValues";

      BOOST_FOREACH(const std::string& attachment_name, attachments_names)
      {
        if (attachment_name.find(possible_name) != 0)
        {
          continue;
        }

        document.get_attachment<std::vector<cv::Mat> >(attachment_name, model_colorValues);
      }
      outputs["model_colorValues"] << model_colorValues;

      return ecto::OK;
    }

  private:
    ecto::spore<std::string> json_feature_params_;
    ecto::spore<std::string> json_descriptor_params_;
    ecto::spore<std::string> object_id_;
    ecto::spore<std::string> json_db_;
    std::vector<cv::Mat> model_colorValues;
};

ECTO_CELL(colorhist_detection, ModelReader, "ModelReader",
          "Compute ColorHist models for a given object")

