
#include "CloudVisualizer.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/time.h>

#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/filters/voxel_grid.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/io.h>
#include <pcl/segmentation/euclidean_cluster_comparator.h>
#include <pcl/segmentation/extract_clusters.h>

#include <limits>

#define NUM_COMMAND_ARGS 1

bool openCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloudOut, const char* fileName);
void pointPickingCallback(const pcl::visualization::PointPickingEvent& event, void* cookie);
void keyboardCallback(const pcl::visualization::KeyboardEvent &event, void* viewer_void);
pcl::ModelCoefficients::Ptr segmentPlane(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloudIn, pcl::PointIndices::Ptr &inliers, double distanceThreshold, int maxIterations);
void removePoints(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloudIn, const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloudOut, const pcl::PointIndices::ConstPtr &inliers);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr copyCloud(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud);

int main(int argc, char** argv) {
    
    if(argc != NUM_COMMAND_ARGS + 1) {
        std::cout << "USAGE: " << argv[0] << " " << "<file_name>" << std::endl;
    }

    char* filename = argv[1];

    // initialize the cloud viewer
    CloudVisualizer CV("Rendering Window");

    // open the point cloud
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
    openCloud(cloud, filename);

    // create the result cloud
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudResult(new pcl::PointCloud<pcl::PointXYZRGBA>);


    // *** Begin Processing image ***


    // the maximum distance allowed from a point to the fitted plane for that point to be considered an inlier
    float distanceThreshold = 0.015;
    int maxIterations = 5000;

    // pcl::PointIndices is a PCL ds used to store a list of indices representing inliers of a point cloud
    // tableInliers is a smart pointer that points to the newly allocated pcl::PointIndices object
    // tableInliers is used to store the indices of points that belong to the table surface plane detected during RANSAC
    pcl::PointIndices::Ptr tableInliers(new pcl::PointIndices);

    // We are recieving smart Ptr to a ModelCoefficients object that hold the equation of our detected plane
    pcl::ModelCoefficients::Ptr tableCoefficients = segmentPlane(cloud, tableInliers, distanceThreshold, maxIterations);
    const float distanceToTable = tableCoefficients->values[3];

    // color and copy points to resultCloud
    for(int i = 0; i < tableInliers->indices.size(); i++) {
        
        // tableInliers->indicies stores indexes that map us to points in cloud
        int index = tableInliers->indices.at(i);

        // color the plane points blue
        cloud->points.at(index).r = 0;
        cloud->points.at(index).g = 0;
        cloud->points.at(index).b = 255;

        // take the identified table plane and move to result cloud
        cloudResult->points.push_back(cloud->points.at(index));
    }


    // *Extract Table Plane*
    // We do not want to detect the same plane again (Less computations for RANSAC)
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudFiltered(new pcl::PointCloud<pcl::PointXYZRGBA>);
    removePoints(cloud, cloudFiltered, tableInliers); // cloudFiltered now contains no table plane

    
    // *Downsampling*
    const float voxelSize = 0.005;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudDownSampled(new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::VoxelGrid<pcl::PointXYZRGBA> voxFilter;
    voxFilter.setInputCloud(cloudFiltered);
    voxFilter.setLeafSize(static_cast<float>(voxelSize), static_cast<float>(voxelSize), static_cast<float>(voxelSize));
    voxFilter.filter(*cloudDownSampled);


    // *Clustering*
    const float clusterDistance = 0.01; // claiming that all boxes we are looking for are a minimum of 1cm apart
    int minClusterSize = 220; // a cluster less than 220 pts wont be returned
    int maxClusterSize = 100000;
    // We have a vector of vectors of indicies. If we find 3 clusters in our pc, cluster indicies will have
    // 3 elements, and each one of those elements is a list of integer indices pointing back to the original cloud
    std::vector<pcl::PointIndices> clusterIndices;
    pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBA>);
    tree->setInputCloud(cloudDownSampled);

    // create the euclidian cluster extraction object
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGBA> ec;
    ec.setClusterTolerance(clusterDistance);
    ec.setMinClusterSize(minClusterSize);
    ec.setMaxClusterSize(maxClusterSize);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloudDownSampled);
    ec.extract(clusterIndices);


    // *Process Clusters*
    // At this point each of our boxes are identified as a cluster within clusterIndices and all noise is filtered out,
    // Iterate through each of our identified clusters
    for(int i = 0; i < clusterIndices.size(); i++) {

        // Select a cluster (represents a box surface)
        pcl::PointIndices::Ptr clusterInliers(new pcl::PointIndices(clusterIndices[i]));

        // The output from detecting a plane
        pcl::PointIndices::Ptr resultantInliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr clusterCoeffs(new pcl::ModelCoefficients);

        // Performinag plane segmentation 
        pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
        seg.setOptimizeCoefficients(true);
        seg.setInputCloud(cloudDownSampled);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(5000);
        seg.setDistanceThreshold(0.0254);
        seg.setIndices(clusterInliers);
        seg.segment(*resultantInliers, *clusterCoeffs);

        // track our x,y min/max for each box surface
        float minX = std::numeric_limits<float>::max(), maxX = std::numeric_limits<float>::lowest();
        float minY = std::numeric_limits<float>::max(), maxY = std::numeric_limits<float>::lowest();
        float planeHeight = clusterCoeffs->values[3];

        // Draw the points of our detected plane onto the resultCloud with appropriate colors
        // and determine min/max x and y values
        for(int j = 0; j < resultantInliers->indices.size(); j++) {

            int index = resultantInliers->indices.at(j);

            // First box will be green, second will be red 
            cloudDownSampled->points.at(index).r = (i == 0 ? 0 : 255);
            cloudDownSampled->points.at(index).g = (i == 0 ? 255 : 0);
            cloudDownSampled->points.at(index).b = 0;

            // Grab the x,y value for the current point
            float currX = cloudDownSampled->points.at(index).x;
            float currY = cloudDownSampled->points.at(index).y;

            // Update largest and smallest points
            if(currX < minX) minX = currX; 
            if(currX > maxX) maxX = currX;
            if(currY < minY) minY = currY;
            if(currY > maxY) maxY = currY;

            // take the identified box plane and move to result cloud
            cloudResult->points.push_back(cloudDownSampled->points.at(index));
            //std::cout << "Point " << j << ": " << cloudDownSampled->points.at(index) << std::endl;

        }

        float length = maxX - minX;
        float width = maxY - minY;
        float height = distanceToTable - planeHeight;

        std::cout << "BOX " << i+1 << ": " << length << " " << width << " " << height << std::endl;
    }

    // Notes on cluster thresholds
    // 1 biggest is 570, smallest is 224, all noise is smaller than our smallest box
    // 2 big: 526, no other noise identified
    // 3 biggest is 536, smallest is 274, all noise is smaller than our smaller box
    // 4 biggest is 376, smallest is 362, all noise is smaller than our smallest box
    // 5 biggest is 404, smallest is 254, all noise is smaller than our smallest box


    // render the scene
    CV.addCloud(cloudResult);
    CV.addCoordinateFrame(cloudResult->sensor_origin_, cloudResult->sensor_orientation_);

    // register mouse and keyboard event callbacks
    CV.registerPointPickingCallback(pointPickingCallback, cloudResult);
    CV.registerKeyboardCallback(keyboardCallback);

    // enter visualization loop
    while(CV.isRunning())
    {
        CV.spin(100);
    }

    // exit program
    return 0;
}

// The func will populate inliers with with the indicies of the inliers found during plane seg allowing us
// to access the points in the original point cloud
pcl::ModelCoefficients::Ptr segmentPlane(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloudIn, pcl::PointIndices::Ptr &inliers, double distanceThreshold, int maxIterations) {

    // Object to store the model coefficients (the plane equation parameters) of the detected plane
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    // Creating seg to setup our parameters
    pcl::SACSegmentation<pcl::PointXYZRGBA> seg;

    // RANSAC will find initial plane model and then further refine the coefficients by performing a least-squares
    // fit using all the inliers. Can help improve accurracy and with the presence of noise or outliers
    seg.setOptimizeCoefficients(true);
    seg.setInputCloud(cloudIn);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(maxIterations);
    seg.setDistanceThreshold(distanceThreshold);

    seg.segment(*inliers, *coefficients);

    
    // return our plane coefficients for box dimensions
    return coefficients;
}

void removePoints(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloudIn, const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloudOut, const pcl::PointIndices::ConstPtr &inliers) {
    
    pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
    extract.setInputCloud(cloudIn);
    extract.setIndices(inliers);
    
    // Filter is discarding the points specified by the indices and is keeping all other points
    // We are removing a detected plane from cloudIn
    extract.setNegative(true);

    // cloudOut now contains all points except the points specified by inliers
    extract.filter(*cloudOut);
}

pcl::PointCloud<pcl::PointXYZRGBA>::Ptr copyCloud(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud) {

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr workingCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
    *workingCloud = *cloud;

    return workingCloud;
}

bool openCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloudOut, const char* fileName) {
    
    // convert the file name to string
    std::string fileNameStr(fileName);

    // handle various file types
    std::string fileExtension = fileNameStr.substr(fileNameStr.find_last_of(".") + 1);
    if(fileExtension.compare("pcd") == 0)
    {
        // attempt to open the file
        if(pcl::io::loadPCDFile<pcl::PointXYZRGBA>(fileNameStr, *cloudOut) == -1)
        {
            PCL_ERROR("error while attempting to read pcd file: %s \n", fileNameStr.c_str());
            return false;
        }
        else
        {
            return true;
        }
    }
    else if(fileExtension.compare("ply") == 0)
    {
        // attempt to open the file
        if(pcl::io::loadPLYFile<pcl::PointXYZRGBA>(fileNameStr, *cloudOut) == -1)
        {
            PCL_ERROR("error while attempting to read pcl file: %s \n", fileNameStr.c_str());
            return false;
        }
        else
        {
            return true;
        }
    }
    else
    {
        PCL_ERROR("error while attempting to read unsupported file: %s \n", fileNameStr.c_str());
        return false;
    }
}

void pointPickingCallback(const pcl::visualization::PointPickingEvent& event, void* cookie)
{
    static int pickCount = 0;
    static pcl::PointXYZRGBA lastPoint;

    pcl::PointXYZRGBA p;
    event.getPoint(p.x, p.y, p.z);

    cout << "POINT CLICKED: " << p.x << " " << p.y << " " << p.z << endl;

    // if we have picked a point previously, compute the distance
    if(pickCount % 2 == 1)
    {
        double d = std::sqrt((p.x - lastPoint.x) * (p.x - lastPoint.x) + (p.y - lastPoint.y) * (p.y - lastPoint.y) + (p.z - lastPoint.z) * (p.z - lastPoint.z));
        cout << "DISTANCE BETWEEN THE POINTS: " << d << endl;
    }

    // update the last point and pick count
    lastPoint.x = p.x;
    lastPoint.y = p.y;
    lastPoint.z = p.z;
    pickCount++;
}

void keyboardCallback(const pcl::visualization::KeyboardEvent &event, void* viewer_void)
{
    // handle key down events
    if(event.keyDown())
    {
        // handle any keys of interest
        switch(event.getKeyCode())
        {
            case 'a':
                cout << "KEYPRESS DETECTED: '" << event.getKeySym() << "'" << endl;
                break;
            default:
                break;
        }
    }
}