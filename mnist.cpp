#include "mnist.h"
int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void create_image(CvSize size, int channels, unsigned char data[28][28], int imagenumber) {
    std::string imgname; std::ostringstream imgstrm; std::string fullpath;
    imgstrm << imagenumber;
    imgname = imgstrm.str();
    fullpath = "DataSet/"+ imgname+".jpg";

    IplImage *imghead=cvCreateImageHeader(size, IPL_DEPTH_8U, channels);
    cvSetData(imghead, data, size.width);
    cvSaveImage(fullpath.c_str(),imghead);
}

void parse_and_save_idx3 ( const char* filename )
{
    std::ifstream file (filename);

    if (file.is_open())
    {
        int magic_number=0; int number_of_images=0;int r; int c;
        int n_rows=0; int n_cols=0;CvSize size;unsigned char temp=0;

        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);

        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);

        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);

        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);

        unsigned char arr[28][28];

        for(int i=0;i<1000;++i)
        {
             for(r=0;r<n_rows;++r)
             {
                for(c=0;c<n_cols;++c)
                 {
                     file.read((char*)&temp,sizeof(temp));
                    arr[r][c]= temp;
                 }
             }
            size.height=r;  size.width=c;
            create_image(size,1,arr, i);
        }
    }/* end of if (file.is_open()) */
    else
    {
        std::cout << "The file can not be opened. Exit.\n";
    } /* end of else */
}

