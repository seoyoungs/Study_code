using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.IO.Ports;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;


namespace TMD2
{
    public partial class pb_PT : Form
    {
        Circle paper1; // 선언만
        SerialPort _uart = new SerialPort();
        Queue<byte> _serialQueue = new Queue<byte>();

        int x = 220;
        int y = 220;
        int radius = 215;

        int small_x = 220;
        int small_y = 220;

        int small_radius = 180;
        int tiny_x = 100;
        int tiny_y = 100;

        int tiny_radius = 5;

        Timer _drawTick = new Timer();
        //타이머 상태체크 변수 추가
        private bool bView = false;

        //float angle = 30;
        //float km = 11;

        // 데이터 추가
        List<Datas> _datasTable = new List<Datas>();

        int angle_count = 0;
        int km_count = 0;
        int[] km_list = new int[] { 2, 2, 2, 2, 2, 2 ,6,6,6,6,6,6 };
        int[] angle_list = new int[] { 30, 110, 150, 210, 250, 320, 30, 110, 150, 210, 250, 320 };

        public pb_PT()
        {
            InitializeComponent();
            paper1 = new Circle(1000, 1000);
            // 첫 시작은 이모티콘 안보이게
            pt_Human.Visible = false;
            pb_Car.Visible = false;
            _drawTick.Interval = 600;
            //_drawTick.Tick += new EventHandler(timer_Tick);
            _drawTick.Tick += (sender, e) =>
            {
                if (_datasTable.Count == 0)//들어온 데이터가 없으면 return
                {
                    return;
                }

                // 리스트 합치거나 별도로 관리해서 뒤에 관리할지(밑에 draw함수에서 매개변수 2개 받으므로 추가)
                List<Datas> drawEntitys1; // list 타입의 entity를 넣는다.
                drawEntitys1 = new List<Datas>(_datasTable);
                timer_Tick(drawEntitys1);
                drawEntitys1.Clear(); //초기화

            };
            //이렇게 했을때의 문제점 (datas로 배열이 되어서 1번밖에 안돌아간다.)
            Datas datas = new Datas(km_list, angle_list); //data를 여기다가 집어넣고 시작

            _datasTable.Add(datas);
        }
        private void btn_Serial_Click(object sender, EventArgs e)
        {
            //사람, 차모양 이모티콘 보여주기(PT, VT)
            pt_Human.Visible = true;
            if (pt_Human.Visible == true)
            {
                pb_Car.BackColor = Color.Transparent;
                pb_Car.Parent = pt_Human;
            }
            //pb_Car.Visible = true;




            paper1.DrawCircle_Big(x, y, radius, Color.Blue);
            paper1.DrawCircle_Small(small_x, small_y, small_radius, Color.Red); // 위험구간(작은원)
            paper1.TinyPie_Fill(tiny_x, tiny_y, tiny_radius, Color.Gray);


            //paper1.DrawPie_Small_0(small_x, small_y, small_radius, Color.Red);
            //paper1.DrawPie_Small_60(small_x, small_y, small_radius, Color.Red);
            //paper1.DrawPie_Small_120(small_x, small_y, small_radius, Color.Red);
            //paper1.DrawPie_Small_180(small_x, small_y, small_radius, Color.Red);
            //paper1.DrawPie_Small_240(small_x, small_y, small_radius, Color.Red);
            //paper1.DrawPie_Small_300(small_x, small_y, small_radius, Color.Red);


            pictureBox1.Image = paper1.GetPaper();


            





            // 선택된 port가 없거나 이미 열려있으면 나가기
            if (cb_Port.SelectedIndex < 0)
            {
                //MessageBox.Show("포트를 연결해 주세요.");
                //return;
            }

            if (_uart.IsOpen)
            {
                //MessageBox.Show("포트가 이미 열려있습니다.");
                //return;
            }

            //_uart = new SerialPort();
            //_uart.DataReceived += new SerialDataReceivedEventHandler(_uart_DataReceived);
            //_uart.PortName = cb_Port.SelectedItem.ToString();
            //_uart.ReadBufferSize = 8; // 버퍼를 8개씩 읽는다. 한꺼번에
            //_uart.BaudRate = 115200;
            //_uart.DataBits = 8;
            //_uart.Parity = Parity.None;
            //_uart.StopBits = StopBits.One;

            //_uart.Open();

            //btn_Clear.Enabled = true;
            //btn_Serial.Enabled = false;


        }

        private void _uart_DataReceived(object sender, SerialDataReceivedEventArgs e)
        {
            int count = _uart.BytesToRead;

            Console.WriteLine($"Input Data Length {count}");
            for (int i = 0; i < count; ++i)
            {

                

                int data = _uart.ReadByte();
                if (data == -1)
                {
                    return;
                }
                if (data == ';')
                {
                    string dataString = "";
                    try
                    {
                        byte[] temp = _serialQueue.ToArray();//queue 담을 그릇
                        _serialQueue.Clear();//초기화
                        dataString = Encoding.ASCII.GetString(temp);

                        if (Int32.Parse(dataString) != 0)
                        {
                            string[] token = dataString.Split(',');// 좌표 , 로 된 것 대로 자르기

                            for (int j = 0; j < token.Length; ++j)
                            {

                                //km = float.Parse(token[j]);
                                //j += 1;// 1을 더해야 x좌표 다음인 y좌표가 있다.
                                //angle = float.Parse(token[j]);

                                // entitys.Add(entity);
                                //Log($"[Parse Entity] X = {entity.X} Y = {entity.Y} Type = {entity.Type.ToString()}");
                            }

                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine("Wrong Format Pass!");
                        continue;
                    }
                }
                else
                {
                    _serialQueue.Enqueue((byte)data);
                }
            }
        }

        private void Test_Click(object sender, EventArgs e)
        {
            angle_count = 0; // 인덱스 초기화
            km_count = 0;
            // 이벤트 생성은 한번만 가능하게 해준다(Main 폼에서 실행)

            _drawTick.Start();
        }

        private void timer_Tick(List<Datas> drawObjects)
        {
            //km_count += 1;
            //angle_count += 1;

            //if (paper1 != null)
            //{
            //    paper1.Clear();
            //    paper1 = null;
            //}

            if (bView == true)
            {
                bView = false;
                paper1.DrawCircle_Big(x, y, radius, Color.Blue);
                paper1.DrawCircle_Small(small_x, small_y, small_radius, Color.Red); // 위험구간(작은원)
                paper1.TinyPie_Fill(tiny_x, tiny_y, tiny_radius, Color.Gray);


                
                //angle_count -= 1;
                //km_count -= 1;
                pictureBox1.Image = paper1.GetPaper();

                //if (angle_list.Length == i || km_list.Length == i)
                //{
                //    angle_count = 0;
                //    km_count = 0;
                //    _drawTick.Stop();
                //}

            }

            else
            {
                bView = true;

                for (int i = 0; i < angle_list.Length; ++i)
                {
                    //for문을 사용해 6번 돌리기

                    if (km_list[i] <= 5) //km > 5 && km <= 10
                    {
                        if (angle_list[i] < 60)
                        {
                            //안에 빨간색 보일때
                            paper1.DrawCircle_Small(small_x, small_y, small_radius, Color.Black); // 위험구간(작은원)
                            paper1.DrawPie_Small_300(small_x, small_y, small_radius, Color.Red); //위험 구간(작은)
                            paper1.TinyPie_Fill(tiny_x, tiny_y, tiny_radius, Color.Gray);
                        }
                        if (angle_list[i] >= 60 && angle_list[i] < 120)
                        {
                            //안에 빨간색 보일때
                            paper1.DrawCircle_Small(small_x, small_y, small_radius, Color.Black); // 위험구간(작은원)
                            paper1.DrawPie_Small_240(small_x, small_y, small_radius, Color.Red); //위험 구간(작은)
                            paper1.TinyPie_Fill(tiny_x, tiny_y, tiny_radius, Color.Black);
                        }
                        if (angle_list[i] >= 120 && angle_list[i] < 180)
                        {
                            //안에 빨간색 보일때
                            paper1.DrawCircle_Small(small_x, small_y, small_radius, Color.Black); // 위험구간(작은원)
                            paper1.DrawPie_Small_180(small_x, small_y, small_radius, Color.Red); //위험 구간(작은)
                            paper1.TinyPie_Fill(tiny_x, tiny_y, tiny_radius, Color.Black);
                        }
                        if (angle_list[i] >= 180 && angle_list[i] < 240)
                        {
                            //안에 빨간색 보일때
                            paper1.DrawCircle_Small(small_x, small_y, small_radius, Color.Black); // 위험구간(작은원)
                            paper1.DrawPie_Small_120(small_x, small_y, small_radius, Color.Red); //위험 구간(작은)
                            paper1.TinyPie_Fill(tiny_x, tiny_y, tiny_radius, Color.Black);
                        }
                        if (angle_list[i] >= 240 && angle_list[i] < 300)
                        {
                            //안에 빨간색 보일때
                            paper1.DrawCircle_Small(small_x, small_y, small_radius, Color.Black); // 위험구간(작은원)
                            paper1.DrawPie_Small_60(small_x, small_y, small_radius, Color.Red); //위험 구간(작은)
                            paper1.TinyPie_Fill(tiny_x, tiny_y, tiny_radius, Color.Black);
                        }
                        if (angle_list[i] >= 300 && angle_list[i] < 360)
                        {
                            //안에 빨간색 보일때
                            paper1.DrawCircle_Small(small_x, small_y, small_radius, Color.Black); // 위험구간(작은원)
                            paper1.DrawPie_Small_0(small_x, small_y, small_radius, Color.Red); //위험 구간(작은)
                            paper1.TinyPie_Fill(tiny_x, tiny_y, tiny_radius, Color.Black);
                        }
                       
                        
                    }
                    else if (km_list[i] > 5) //km <= 5
                    {
                        //bView = true;
                        if (angle_list[i] < 60)
                        {
                            //바깥 노란색 보일때
                            paper1.DrawPie_Big_300(x, y, radius, Color.Yellow); //주의 구간(큰원)
                            paper1.DrawPie_Gray_300(small_x, small_y, small_radius, Color.Gray); // 위험구간(작은원 -- 채우기)
                            paper1.DrawCircle_Small(small_x, small_y, small_radius, Color.Black); // 위험구간(작은원)
                            paper1.TinyPie_Fill(tiny_x, tiny_y, tiny_radius, Color.Black);
                        }
                        if (angle_list[i] >= 60 && angle_list[i] < 120)
                        {
                            //바깥 노란색 보일때
                            paper1.DrawPie_Big_240(x, y, radius, Color.Yellow); //주의 구간(큰원)
                            paper1.DrawGray_Small_240(small_x, small_y, small_radius, Color.Gray); // 위험구간(작은원 -- 채우기)
                            paper1.DrawCircle_Small(small_x, small_y, small_radius, Color.Black); // 위험구간(작은원)
                            paper1.TinyPie_Fill(tiny_x, tiny_y, tiny_radius, Color.Black);
                        }
                        if (angle_list[i] >= 120 && angle_list[i] < 180)
                        {
                            //바깥 노란색 보일때
                            paper1.DrawPie_Big_180(x, y, radius, Color.Yellow); //주의 구간(큰원)
                            paper1.DrawGray_Small_180(small_x, small_y, small_radius, Color.Gray); // 위험구간(작은원 -- 채우기)
                            paper1.DrawCircle_Small(small_x, small_y, small_radius, Color.Black); // 위험구간(작은원)
                            paper1.TinyPie_Fill(tiny_x, tiny_y, tiny_radius, Color.Black);
                        }
                        if (angle_list[i] >= 180 && angle_list[i] < 240)
                        {
                            //바깥 노란색 보일때
                            paper1.DrawPie_Big_120(x, y, radius, Color.Yellow); //주의 구간(큰원)
                            paper1.DrawGray_Small_120(small_x, small_y, small_radius, Color.Gray); // 위험구간(작은원 -- 채우기)
                            paper1.DrawCircle_Small(small_x, small_y, small_radius, Color.Black); // 위험구간(작은원)
                            paper1.TinyPie_Fill(tiny_x, tiny_y, tiny_radius, Color.Black);
                        }
                        if (angle_list[i] >= 240 && angle_list[i] < 300)
                        {
                            //바깥 노란색 보일때
                            paper1.DrawPie_Big_60(x, y, radius, Color.Yellow); //주의 구간(큰원)
                            paper1.DrawGray_Small_60(small_x, small_y, small_radius, Color.Gray); // 위험구간(작은원 -- 채우기)
                            paper1.DrawCircle_Small(small_x, small_y, small_radius, Color.Black); // 위험구간(작은원)
                            paper1.TinyPie_Fill(tiny_x, tiny_y, tiny_radius, Color.Black);
                        }
                        if (angle_list[i] >= 300 && angle_list[i] < 360)
                        {
                            //바깥 노란색 보일때
                            paper1.DrawPie_Big_0(x, y, radius, Color.Yellow); //주의 구간(큰원)
                            paper1.DrawGray_Small_0(small_x, small_y, small_radius, Color.Gray); // 위험구간(작은원 -- 채우기)
                            paper1.DrawCircle_Small(small_x, small_y, small_radius, Color.Black); // 위험구간(작은원)
                            paper1.TinyPie_Fill(tiny_x, tiny_y, tiny_radius, Color.Black);
                        }


                    }
                    //else if (km > 10)
                    //{
                    //    MessageBox.Show($"{km}으로 거리를 벗어났습니다.");
                    //    _drawTick.Stop();
                    //}
                    pictureBox1.Image = paper1.GetPaper();
                }
                //

            }
            
                
            
        }

        private void Clear_Click(object sender, EventArgs e)
        {
            _drawTick.Stop();
            paper1.Clear();
            pictureBox1.Image = paper1.GetPaper();
            pt_Human.Visible = false;
            pb_Car.Visible = false;

            if (cb_Port.SelectedIndex < 0)
            {
                return;
            }
            try
            {
                _uart.Close();

                MessageBox.Show("포트 Close Successed");
            }
            catch (Exception ex)
            {
                MessageBox.Show("포트 Close Failed");
            }
            finally
            {
                btn_Clear.Enabled = false;
                btn_Serial.Enabled = true;
            }
        }
    }
}
