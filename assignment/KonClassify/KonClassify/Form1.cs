using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace KonClassify
{
    public partial class Form1 : Form
    {
        private const int imageSize = 227;
        private Model.Kon model;

        public Form1()
        {
            InitializeComponent();
        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void Form1_Load(object sender, EventArgs e)
        {
            model = new Model.Kon();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            bool isSuccess = false;

            label1.Text = string.Empty;
            pictureBox1.Image = null;
            pictureBox1.Refresh();

            //Load Image
            try
            {
                pictureBox1.Load(textBox1.Text);
                isSuccess = true;
            }
            catch(Exception ex)
            {
                MessageBox.Show("图片读取错误！",ex.Message);
            }

            if (isSuccess)
            {
                //Reform the input image
                Bitmap clonedBmp = new Bitmap(imageSize, imageSize);
                Graphics gNormalized = Graphics.FromImage(clonedBmp);

                gNormalized.DrawImage(pictureBox1.Image, 0, 0, imageSize, imageSize);

                var imageArray = new float[imageSize * imageSize * 3];
                for(int y = 0; y < imageSize; y++)
                {
                    for(int x = 0;x < imageSize; x++)
                    {
                        var color = clonedBmp.GetPixel(x, y);

                        imageArray[y * imageSize + x] = color.B;
                        imageArray[y * imageSize + x + imageSize * imageSize] = color.G;
                        imageArray[y * imageSize + x + 2 * imageSize * imageSize] = color.R;
                    }
                }

                //Get result by accessing model
                var result = model.Infer(new List<IEnumerable<float>> { imageArray }).First().First();

                label1.Text = result;
            }

        }
    }
}
