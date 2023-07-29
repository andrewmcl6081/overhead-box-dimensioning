
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

#define NUM_COMMAND_ARGS 1

bool openCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloudOut, const char* fileName);
void pointPickingCallback(const pcl::visualization::PointPickingEvent& event, void* cookie);
void keyboardCallback(const pcl::visualization::KeyboardEvent &event, void* viewer_void);
pcl::ModelCoefficients::Ptr segmentPlane(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &cloudIn, pcl::PointIndices::Ptr &inliers, double distanceThreshold, int maxIterations);
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


    // segment the table surface
    float distanceThreshold = 0.015; // the maximum distance allowed from a point to the fitted plane for that point to be considered an inlier
    int maxIterations = 5000;

    // pcl::PointIndices is a PCL ds used to store a list of indices representing inliers of a point cloud
    // inliers is a smart pointer that points to the newly allocated pcl::PointIndices object
    // inliers is used to store the indices of points that belong to a particular plane detected during plane seg using RANSAC

    pcl::PointIndices::Ptr tableInliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr tableCoefficients = segmentPlane(cloud, tableInliers, distanceThreshold, maxIterations);
    const float tableDistance = tableCoefficients->values[3];

    std::cout << "Seg Results: " << tableInliers->indices.size() << " points" << std::endl;
    std::cout << "Camera to Table Distance = " << tableDistance << std::endl;

    for(int i = 0; i < tableInliers->indices.size(); i++) {
        
        // retrieving the index of the current inlier point. This index corresponds to a specific
        // point in the original point cloud
        int index = tableInliers->indices.at(i);

        // make the table plane blue in our original cloud
        cloud->points.at(index).r = 0;
        cloud->points.at(index).g = 0;
        cloud->points.at(index).b = 255;

        // take the identified table plane and move to result cloud
        cloudResult->points.push_back(cloud->points.at(index));
    }


    // remove the table plane points so we do not detect the same plane again. Less computations for RANSAC
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudFiltered(new pcl::PointCloud<pcl::PointXYZRGBA>);
    removePoints(cloud, cloudFiltered, tableInliers);


    // down sample before clustering
    const float voxelSize = 0.005;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudDownSampled(new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::VoxelGrid<pcl::PointXYZRGBA> voxFilter;
    voxFilter.setInputCloud(cloudFiltered);
    voxFilter.setLeafSize(static_cast<float>(voxelSize), static_cast<float>(voxelSize), static_cast<float>(voxelSize));
    voxFilter.filter(*cloudDownSampled);
    std::cout << "Points before: " << cloudFiltered->points.size() << std::endl;
    std::cout << "Points after: " << cloudDownSampled->points.size() << std::endl;

    
    const float clusterDistance = 0.01; // claiming that all boxes we are looking for are a minimum of 1cm apart
    int minClusterSize = 220; // a cluster less than 220 pts wont be returned
    int maxClusterSize = 100000;


    // we have a vector of vectors of indicies. If we find 3 clusters in our pc, cluster indicies will have
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

    // perform the clustering on the cloud that is filtered
    ec.extract(clusterIndices);
    std::cout << "Clusters identified: " << clusterIndices.size() << std::endl;


    for(int i = 0; i < clusterIndices.size(); i++) {
        int r,g,b;

        if(i == 0) {
            r = 0;
            g = 0;
            b = 255;
        }
        else if(i == 1){
            r = 255;
            g = 255;
            b = 255;
        }
        else if(i == 2) {
            r = 0;
            g = 255;
            b = 0;
        }
        else{
            r = 255;
            g = 0;
            b = 0;
        }
        std::cout << "Cluster " << i << " size: " << clusterIndices.at(i).indices.size() << ", color: " << r << " " << g << " " << b << std::endl;

        // for each point within a specific cluster, we are giving it a random color
        for(int j = 0; j < clusterIndices.at(i).indices.size(); j++) {
            cloudDownSampled->points.at(clusterIndices.at(i).indices.at(j)).r = r;
            cloudDownSampled->points.at(clusterIndices.at(i).indices.at(j)).g = g;
            cloudDownSampled->points.at(clusterIndices.at(i).indices.at(j)).b = b;
        }
    }

    // At this point each of our boxes are identified as a cluster and all noise is filtered out,
    // time to fit them to a plane, and calculate their L, W, H

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudTemp(new pcl::PointCloud<pcl::PointXYZRGBA>);

    // fit planes to each of our box surfaces and calculate dimensions
    for(int i = 0; i < clusterIndices.size(); i++) {

        // select a cluster that resembles a box surface
        pcl::PointIndices::Ptr clusterInliersInput(new pcl::PointIndices(clusterIndices[i]));

        // the output from detecting a plane
        pcl::PointIndices::Ptr clusterInliersOutput(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr clusterCoeffs(new pcl::ModelCoefficients);

        pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setInputCloud(cloudDownSampled);
        seg.setMaxIterations(5000);
        seg.setDistanceThreshold(0.0254);
        seg.setIndices(clusterInliersInput);
        seg.segment(*clusterInliersOutput, *clusterCoeffs);

        int r,g,b;

        // draw the points of our detected plane onto the result
        for(int j = 0; j < clusterInliersInput->indices.size(); j++) {
            int index = clusterInliersInput->indices.at(j);

            if(i == 0) {
                r = 0;
                g = 255;
                b = 0;
            }
            else {
                r = 255;
                g = 0;
                b = 0;
            }

            cloudDownSampled->points.at(index).r = r;
            cloudDownSampled->points.at(index).g = g;
            cloudDownSampled->points.at(index).b = b;

            // take the identified box plane and move to result cloud
            cloudResult->points.push_back(cloudDownSampled->points.at(index));
        }

        std::cout << '\n' << std::endl;
        std::cout << "cluster indices size: " << clusterIndices.size() << std::endl;
        std::cout << "cluster color: " << r << ", " << g << ", " << b << std::endl;
        std::cout << "clusterInliersInput size: " << clusterInliersInput->indices.size() << std::endl;
        std::cout << "clusterInliersOutput size: " << clusterInliersOutput->indices.size() << std::endl;
        std::cout << "Height: " << tableDistance - clusterCoeffs->values[3] << std::endl; 
        std::cout << '\n' << std::endl;
    }

    
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
// to access the inlier points in the original point cloud
pcl::ModelCoefficients::Ptr segmentPlane(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &cloudIn, pcl::PointIndices::Ptr &inliers, double distanceThreshold, int maxIterations) {

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
    // true = ExtractIndicies will extract the points that are Not part of the inliers
    // The extracted points will be those outside the detected plane
    extract.setNegative(true);
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