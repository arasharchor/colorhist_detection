#include <ecto/ecto.hpp>
#include <string>
#include <map>
#include <vector>

#include <boost/foreach.hpp>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/flann/flann.hpp>

#include <object_recognition_core/common/json_spirit/json_spirit.h>
#include <object_recognition_core/common/types.h>
#include <object_recognition_core/db/ModelReader.h>
#include <object_recognition_core/db/opencv.h>

using object_recognition_core::db::Documents;
using object_recognition_core::db::ObjectId;

namespace colorhist
{
  struct DescriptorMatcher: public object_recognition_core::db::bases::ModelReaderBase
  {
    void parameter_callback(const Documents & db_documents)
    {
      colorValues_db_.resize(db_documents.size());
      object_ids_.resize(db_documents.size());

      // Re-load the data from the DB
      std::cout << "Loading models. This may take some time..." << std::endl;
      unsigned int index = 0;
      BOOST_FOREACH(const object_recognition_core::db::Document & document, db_documents)
      {
        ObjectId object_id = document.get_field<std::string>("object_id");
        std::cout << "Loading model for object id: " << object_id << std::endl;
        
        cv::Mat colorValues;
        document.get_attachment<cv::Mat>("colorValues", colorValues);
        colorValues_db_[index] = colorValues;

        // Store the id conversion
        object_ids_[index] = object_id;

        // Compute the span of the object
        float max_span_sq = 0;

        spans_[object_id] = std::sqrt(max_span_sq);
        std::cout << "span: " << spans_[object_id] << " meters" << std::endl;
        ++index;
      }

      // Clear the matcher and re-train it
      matcher_->clear();
      matcher_->add(colorValues_db_);
    }

    static void declare_params(ecto::tendrils& p)
    {
      object_recognition_core::db::bases::declare_params_impl(p, "ColorHist");
    }

    static void declare_io(const ecto::tendrils& params, ecto::tendrils& inputs, ecto::tendrils& outputs)
    {
      inputs.declare < cv::Mat > ("colorValues", "Color values in a matrix");
      
      outputs.declare < std::vector<ObjectId> > ("object_ids", "The ids of the objects");
      outputs.declare < std::map<ObjectId, float> > ("spans", "The ids of the objects");
    }

    void configure(const ecto::tendrils& params, const ecto::tendrils& inputs, const ecto::tendrils& outputs)
    {
      configure_impl();
      // get some parameters
      {
        or_json::mObject search_param_tree;
        std::stringstream ssparams;
        ssparams << params.get < std::string > ("search_json_params");

        {
          or_json::mValue value;
          or_json::read(ssparams, value);
          search_param_tree = value.get_obj();
        }
      }
    }

    /** Get the 2d keypoints and figure out their 3D position from the depth map
     * @param inputs
     * @param outputs
     * @return
     */
    int process(const ecto::tendrils& inputs, const ecto::tendrils& outputs)
    {
      std::vector < std::vector<cv::DMatch> > matches;
      const cv::Mat & colorValues = inputs.get < cv::Mat > ("colorValues");

      outputs["object_ids"] << object_ids_;
      outputs["spans"] << spans_;

      return ecto::OK;
    }
  private:
    /** The object used to match descriptors to our DB of descriptors */
    cv::Ptr<cv::DescriptorMatcher> matcher_;
    /** The descriptors loaded from the DB */
    std::vector<cv::Mat> colorValues_db_;
    /** For each object id, the maximum distance between the known descriptors (span) */
    std::map<ObjectId, float> spans_;
  };
}

ECTO_CELL(colorhist_detection, colorhist::DescriptorMatcher, "DescriptorMatcher",
          "Given descriptors, find matches, relating to objects.");
