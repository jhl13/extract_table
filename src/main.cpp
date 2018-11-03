#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
    Mat src = imread("/home/luo/workspace/extract_table/bin/table1");

    // Check if image is loaded fine
    if (!src.data)
    {
        cerr << "Problem loading image!!!" << endl;
        exit(1);
    }

    // resizing for practical reasons
    Mat rsz;
    Size size(900, 1000);
    resize(src, rsz, size);

    imshow("rsz", rsz); //原始图像

    // Transform source image to gray if it is not
    Mat gray;

    if (rsz.channels() == 3)
    {
        cvtColor(rsz, gray, CV_BGR2GRAY);
    }
    else if (rsz.channels() == 1)
    {
        gray = rsz;
    }
    else
    {
        cout << "It is not a normal form!" << endl;
        exit(1);
    }

    // Show gray image
    imshow("gray", gray);

    Mat gauss_gray;
    GaussianBlur(gray, gauss_gray, Size(5, 5), 0, 0); //高斯滤波去除噪声

    // Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    Mat bw;
    adaptiveThreshold(~gauss_gray, bw, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
    // Show binary image
    imshow("binary", bw);
    vector<Vec4i> hierarchy_transform;
    std::vector<std::vector<cv::Point>> contours_transform;
    cv::findContours(bw, contours_transform, hierarchy_transform, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    vector<vector<Point2f>> contours_poly_transform(contours_transform.size());

    for (int num = 0; num < contours_transform.size(); num++)
    {
        //进行透视变换
        approxPolyDP(Mat(contours_transform[num]), contours_poly_transform[num], 10, true); //拟合四边形
        if (double(contourArea(contours_transform[num])) < 3000 || (contours_poly_transform[num].size() != 4))
            continue;
        stringstream s_num;
        string pic_num;
        s_num << num;
        s_num >> pic_num;

        Mat transform_img, transform_img_bw, transform_rsz;
        Mat resultImgCorners(4, 1, CV_32FC2);
        resultImgCorners.ptr<Point2f>(0)[0] = Point2f(0, 0);
        resultImgCorners.ptr<Point2f>(0)[1] = Point2f(0, 1000);
        resultImgCorners.ptr<Point2f>(0)[2] =
            Point2f(900, 1000);
        resultImgCorners.ptr<Point2f>(0)[3] = Point2f(900, 0);
        cout << "contours_poly_transform[num]" << contours_poly_transform[num] << endl;
        cout << "resultImgCorners" << resultImgCorners << endl;
        Mat transformation = getPerspectiveTransform(contours_poly_transform[num], resultImgCorners);
        warpPerspective(gray, transform_img, transformation, Size(900, 1000),
                        INTER_NEAREST); //灰度图透视变换

        warpPerspective(bw, transform_img_bw, transformation, Size(900, 1000),
                        INTER_NEAREST); //二值图透视变换

        warpPerspective(rsz, transform_rsz, transformation, Size(900, 1000),
                        INTER_NEAREST); //原图透视变换

        // rsz = transform_rsz;

        for (int j = 0; j <= 3; j++)
        {
            line(transform_img, resultImgCorners.ptr<Point2f>(0)[j], resultImgCorners.ptr<Point2f>(0)[(j + 1) % 4], Scalar(0), 5);
        }
        threshold(~transform_img, transform_img, 150, 255, THRESH_BINARY);
        bitwise_or(transform_img, transform_img_bw, transform_img);
        imshow(pic_num + "transform_img", transform_img);
        // imshow(pic_num + "transform_img_bw", transform_img_bw);
        // imshow(pic_num + "transform_rsz", transform_rsz);
        // Create the images that will use to extract the horizonta and vertical lines
        Mat horizontal = transform_img.clone();
        Mat vertical = transform_img.clone();

        int scale = 15; // play with this variable in order to increase/decrease the amount of lines to be detected

        // Specify size on horizontal axis
        int horizontalsize = horizontal.cols / scale;

        // Create structure element for extracting horizontal lines through morphology operations
        Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize, 1));

        // Apply morphology operations
        erode(horizontal, horizontal, horizontalStructure, Point(-1, -1)); //减少噪声
        dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1));

        // Show extracted horizontal lines
        imshow(pic_num + "horizontal", horizontal);
        // Specify size on vertical axis
        int verticalsize = vertical.rows / scale;

        // Create structure element for extracting vertical lines through morphology operations
        Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1, verticalsize));

        // Apply morphology operations
        erode(vertical, vertical, verticalStructure, Point(-1, -1));
        dilate(vertical, vertical, verticalStructure, Point(-1, -1));
        
        // Show extracted vertical lines
        imshow(pic_num + "vertical", vertical);
        // create a mask which includes the tables
        Mat mask = horizontal + vertical;
        imshow(pic_num + "mask", mask);

        // find the joints between the lines of the tables, we will use this information in order to descriminate tables from pictures (tables will contain more than 4 joints while a picture only 4 (i.e. at the corners))
        Mat joints;
        bitwise_and(horizontal, vertical, joints);
        imshow(pic_num + "joints", joints);
        // Find external contours from the mask, which most probably will belong to tables or to images
        vector<Vec4i> hierarchy;
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
        // RETR_LIST 所有轮廓
        // CV_RETR_EXTERNAL 最外层轮廓
        // CHAIN_APPROX_SIMPLE 仅保留端点
        vector<vector<Point>> contours_poly(contours.size());
        vector<Rect> boundRect_vector;
        vector<Rect> boundRect(contours.size());
        vector<Mat> rois;

        for (size_t i = 0; i < contours.size(); i++)
        {
            // find the area of each contour
            double area = contourArea(contours[i]);

            // filter individual lines of blobs that might exist and they do not represent a table
            if (area < 100) // value is randomly chosen, you will need to find that by yourself with trial and error procedure
                continue;

            approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
            boundRect[i] = boundingRect(Mat(contours_poly[i]));

            // find the number of joints that each table has
            Mat roi = joints(boundRect[i]);

            vector<vector<Point>> joints_contours;
            findContours(roi, joints_contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

            // if the number is not more than 5 then most likely it not a table
            if (joints_contours.size() <= 4)
                continue;

            rois.push_back(transform_rsz(boundRect[i]).clone());
            boundRect_vector.push_back(boundRect[i]);
            //        drawContours( rsz, contours, i, Scalar(0, 0, 255), CV_FILLED, 8, vector<Vec4i>(), 0, Point() );
            rectangle(transform_rsz, boundRect[i].tl(), boundRect[i].br(), Scalar(255, 0, 255), 2, 8, 0);
        }

        for (size_t i = 0; i < rois.size(); i++)
        {
            Mat roi_inner = mask(boundRect[i]);
            imshow(pic_num + "bound", roi_inner);

            Mat image_contour_all = roi_inner.clone();
            Mat image_contour_outside = roi_inner.clone();

            vector<vector<Point>> contours_out;
            vector<Vec4i> hierarchy_out;
            findContours(image_contour_outside, contours_out, hierarchy_out, RETR_EXTERNAL, CHAIN_APPROX_NONE);

            vector<vector<Point>> contours_all;
            vector<Vec4i> hierarchy_all;
            findContours(image_contour_all, contours_all, hierarchy_all, RETR_TREE, CHAIN_APPROX_NONE);
            cout << "size:" << contours_out.size() << endl;
            cout << "size:" << contours_all.size() << endl;
            if (contours_all.size() == contours_out.size())
                return -1; //没有内轮廓，则提前返回

            for (int i = 0; i < contours_out.size(); i++)
            {
                int conloursize = contours_out[i].size();
                for (int j = 0; j < contours_all.size(); j++)
                {
                    if (double(contourArea(contours_all[j])) < 100)
                        contours_all.pop_back();
                    int tem_size = contours_all[j].size();
                    if (conloursize == tem_size)
                    {
                        swap(contours_all[j], contours_all[contours_all.size() - 1]);
                        contours_all.pop_back();
                        break;
                    }
                }
            }

            Mat erodemask = getStructuringElement(MORPH_RECT, Size(3, 3));

            vector<vector<Point>> contours_poly_inner(contours_all.size());
            vector<Rect> boundRect_inner(contours_all.size());
            for (int i = 0; i < contours_all.size(); i++)
            {
                approxPolyDP(Mat(contours_all[i]), contours_poly_inner[i], 3, true);
                if (contours_poly_inner[i].size() < 4 || double(contourArea(contours_all[i]) < 10000))
                    continue;
                cout << "contours_poly_inner:" << contours_poly_inner[i] << endl;

                boundRect_inner[i] = boundingRect(Mat(contours_poly_inner[i]));
                rectangle(transform_rsz, boundRect_inner[i].tl(), boundRect_inner[i].br(), Scalar(255, 255, 0), 2, 8, 0);
            }
        }

        // for (size_t i = 0; i < rois.size(); ++i)
        // {
        //     imshow(pic_num + "roi", rois[i]);
        //     waitKey(1);
        // }
        imshow(pic_num + "contours", transform_rsz);
    }

    waitKey(0);
    return 0;
}