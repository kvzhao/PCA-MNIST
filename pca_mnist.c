#include <highgui.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {

    if ( argc < 1 ) {
        printf("arg");
        return -1;
    }

    int camIndex = atoi(argv[1]) ;

    cvNamedWindow("WebCam",CV_WINDOW_AUTOSIZE);

    CvCapture   *capture = cvCreateCameraCapture( camIndex );
    cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH, 640 );
    cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT, 480 );

    IplImage    *frame;

    while (1) {
        frame = cvQueryFrame( capture );

        if (!frame) break;

        cvShowImage("WebCam", frame);

        char c = cvWaitKey(33);
        if ( c == 27 ) break;
    }

    cvReleaseCapture( &capture );
    cvDestroyWindow ( "WebCam" );

    return 0;
}
