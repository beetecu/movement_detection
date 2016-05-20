/*
* 2014-2015 eliza.glez@gmail.com
*
* OpenCV movement detection
* This is an example about how to detect movement based
* on background substraction.
*
*/

#include "stdafx.h"


#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <iostream>





using namespace std;
using namespace cv;

static void help()
{
	printf("\nDo movement detection.\n"
		"Segment the scene  based on background subtraction.\n"
		"Usage: \n"
		"            ./movement_detection[--camera]=<use camera, if this key is present>, [--file_name]=<path to movie file> \n\n");
}

const char* keys =
{
	"{c  camera   |         | use camera or not}"
	"{m  method   |knn      | method (knn or mog2) }"
	"{fn file_name|C:/tmp/vid2.mpeg | movie file        }"
};

static String Legende(SimpleBlobDetector::Params &pAct)
{
	String s = "";
	if (pAct.filterByArea)
	{
		String inf = static_cast<ostringstream*>(&(ostringstream() << pAct.minArea))->str();
		String sup = static_cast<ostringstream*>(&(ostringstream() << pAct.maxArea))->str();
		s = " Area range [" + inf + " to  " + sup + "]";
	}
	if (pAct.filterByCircularity)
	{
		String inf = static_cast<ostringstream*>(&(ostringstream() << pAct.minCircularity))->str();
		String sup = static_cast<ostringstream*>(&(ostringstream() << pAct.maxCircularity))->str();
		if (s.length() == 0)
			s = " Circularity range [" + inf + " to  " + sup + "]";
		else
			s += " AND Circularity range [" + inf + " to  " + sup + "]";
	}
	if (pAct.filterByColor)
	{
		String inf = static_cast<ostringstream*>(&(ostringstream() << (int)pAct.blobColor))->str();
		if (s.length() == 0)
			s = " Blob color " + inf;
		else
			s += " AND Blob color " + inf;
	}
	if (pAct.filterByConvexity)
	{
		String inf = static_cast<ostringstream*>(&(ostringstream() << pAct.minConvexity))->str();
		String sup = static_cast<ostringstream*>(&(ostringstream() << pAct.maxConvexity))->str();
		if (s.length() == 0)
			s = " Convexity range[" + inf + " to  " + sup + "]";
		else
			s += " AND  Convexity range[" + inf + " to  " + sup + "]";
	}
	if (pAct.filterByInertia)
	{
		String inf = static_cast<ostringstream*>(&(ostringstream() << pAct.minInertiaRatio))->str();
		String sup = static_cast<ostringstream*>(&(ostringstream() << pAct.maxInertiaRatio))->str();
		if (s.length() == 0)
			s = " Inertia ratio range [" + inf + " to  " + sup + "]";
		else
			s += " AND  Inertia ratio range [" + inf + " to  " + sup + "]";
	}
	return s;
}


//this is a sample for foreground detection functions
int main(int argc, const char** argv)
{


	help();

	CommandLineParser parser(argc, argv, keys);
	bool useCamera = parser.has("camera");
	string file = parser.get<string>("file_name");
	string method = parser.get<string>("method");

	bool smoothMask = true;


	/*****************************************************************************/
	//Blob detector initialization
	SimpleBlobDetector::Params pDefaultBLOB;
	// This is default parameters for SimpleBlobDetector
	pDefaultBLOB.thresholdStep = 10;
	pDefaultBLOB.minThreshold = 10;
	pDefaultBLOB.maxThreshold = 220;
	pDefaultBLOB.minRepeatability = 2;
	pDefaultBLOB.minDistBetweenBlobs = 10;
	pDefaultBLOB.filterByColor = false;
	pDefaultBLOB.blobColor = 0;
	pDefaultBLOB.filterByArea = true;  // Region variation filtering by area
	pDefaultBLOB.minArea = 100;
	pDefaultBLOB.maxArea = 15500;
	pDefaultBLOB.filterByCircularity = false;
	pDefaultBLOB.minCircularity = 0.9f;
	pDefaultBLOB.maxCircularity = (float)1e37;
	pDefaultBLOB.filterByInertia = false;
	pDefaultBLOB.minInertiaRatio = 0.01f;
	pDefaultBLOB.maxInertiaRatio = (float)1e37;
	pDefaultBLOB.filterByConvexity = false;
	pDefaultBLOB.minConvexity = 0.95f;
	pDefaultBLOB.maxConvexity = (float)1e37;
	// Descriptor array for BLOB
	vector<String> typeDesc;
	// Param array for BLOB
	vector<SimpleBlobDetector::Params> pBLOB;
	vector<SimpleBlobDetector::Params>::iterator itBLOB;
	// Color palette
	vector< Vec3b >  palette;
	for (int i = 0; i < 65536; i++)
	{
		palette.push_back(Vec3b((uchar)rand(), (uchar)rand(), (uchar)rand()));
	}

	// This descriptor are going to be detect and compute BLOBS with 6 differents params
	// Param for first BLOB detector we want all
	typeDesc.push_back("BLOB");    // see http://docs.opencv.org/trunk/d0/d7a/classcv_1_1SimpleBlobDetector.html
	pBLOB.push_back(pDefaultBLOB);


	itBLOB = pBLOB.begin();
	vector<double> desMethCmp;
	Ptr<Feature2D> b;
	String label;

	// Descriptor loop
	vector<String>::iterator itDesc;
	for (itDesc = typeDesc.begin(); itDesc != typeDesc.end(); itDesc++)
	{
		vector<KeyPoint> keyImg1;
		if (*itDesc == "BLOB")
		{
			b = SimpleBlobDetector::create(*itBLOB);
			label = Legende(*itBLOB);
			itBLOB++;
		}
	}
	vector<KeyPoint>  keyImg;
	vector<Rect>  zone;
	vector<vector <Point> >  region;
	Ptr<SimpleBlobDetector> sbd = b.dynamicCast<SimpleBlobDetector>();

	//Blob initialization end
	/*****************************************************************************/

	VideoCapture cap;
	bool update_bg_model = false;

	if (useCamera)
		cap.open(0);
	else
		cap.open(file.c_str());



	if (!cap.isOpened())
	{
		printf("can not open camera or video file\n");
		return -1;
	}

	namedWindow("image", WINDOW_NORMAL);
	namedWindow("foreground image", WINDOW_NORMAL);

	// CUDA background Substractor initialization
	Ptr<BackgroundSubtractor> bg_model = method == "knn" ?
		createBackgroundSubtractorKNN().dynamicCast<BackgroundSubtractor>() :
		createBackgroundSubtractorMOG2().dynamicCast<BackgroundSubtractor>();

	Mat img0, img, fgmask, fgimg;
	Mat bgimg;

	//vector<Point2f> points[2];

	TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
	Size subPixWinSize(10, 10), winSize(31, 31);

	const int MAX_COUNT = 500;
	bool needToInit = true;
	bool nightMode = false;
	Point2f point;
	bool addRemovePt = false;


	Mat vstatus_mat, verror_mat;

	// MOVEMENT DETECTION LOOP
	for (;;)
	{
		try {


			cap >> img0;

			if (img0.empty())
				break;

			resize(img0, img, Size(640, 640 * img0.rows / img0.cols), INTER_LINEAR);

			if (fgimg.empty())
				fgimg.create(img.size(), img.type());

			//update the model
			bg_model->apply(img, fgmask, update_bg_model ? -1 : 0);
			if (smoothMask)
			{
				GaussianBlur(fgmask, fgmask, Size(11, 11), 3.5, 3.5);
				threshold(fgmask, fgmask, 10, 255, THRESH_BINARY);
			}

			fgimg = Scalar::all(0);
			img.copyTo(fgimg, fgmask);


			bg_model->getBackgroundImage(bgimg);

			// We can detect keypoint with blob detect method

			//Mat     desc, result(fgmask.rows, fgmask.cols, CV_8UC3);
			if (b.dynamicCast<SimpleBlobDetector>() != NULL)
			{

				sbd->detect(fgmask, keyImg, Mat());
				//drawKeypoints(fgmask, keyImg, result);


				int i = 0;
				for (vector<KeyPoint>::iterator k = keyImg.begin(); k != keyImg.end(); k++, i++)
				{
					circle(fgimg, k->pt, (int)k->size, palette[i], 3);
					circle(img, k->pt, (int)k->size, palette[i], 3);
					//cout << "object cooordenates : " << k->pt.x << ";" << k->pt.y << "\n";

					int fontFace = CV_FONT_HERSHEY_COMPLEX;
					double fontScale = 0.4;
					int thickness = 1;

					putText(fgimg, to_string(i + 1), k->pt, fontFace, fontScale,
						palette[i], thickness, 8);
					putText(img, to_string(i + 1), k->pt, fontFace, fontScale,
						palette[i], thickness, 8);

					//putText(fgimg, "object " + to_string(i) + ": " + to_string((int)k->pt.x) + ";" + to_string((int)k->pt.y), Point(15, 10 + 15 * i), fontFace, fontScale,
					//	palette[i], thickness, 8);
				}

				keyImg.clear();
			}



			//namedWindow(*itDesc + label, WINDOW_AUTOSIZE);
			//imshow(*itDesc + label, result);
			//imshow("Original", img);
			//waitKey();

			//imshow("image result", result);

			imshow("image", img);

			//imshow("foreground mask", fgmask);
			imshow("foreground image", fgimg);


			char k = (char)waitKey(1);
			if (k == 27) break;
			if (k == ' ')
			{
				update_bg_model = !update_bg_model;
				if (update_bg_model)
					printf("Background update is on\n");
				else
					printf("Background update is off\n");
			}

		}
		catch (const std::exception&)
		{


			std::cout << "Error, check input file/device" << std::endl;



		}
	}


	cv::destroyAllWindows();
	img0.release();
	img.release();
	fgimg.release();
	bgimg.release();
	keyImg.empty();
	sbd->empty();
	return EXIT_SUCCESS;


}

