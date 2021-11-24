using System;
using System.Text.RegularExpressions;
using static System.Console;

namespace _1_5_example2
{
    class Program
    {
        static void Main(string[] args)
        {
            //a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z
            //A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z
            
            while (true)
            {
                Console.WriteLine("값을 입력해주세요");
                string input = Console.ReadLine();
                string k = input.Substring(0, 1); // 앞자리 숫자만 오게(다른 방법)


                if (Int32.TryParse(k, out int numValue))
                {
                    //Console.WriteLine(numValue);
                }
                else
                {
                    Console.WriteLine($"Int32.TryParse could not parse '{k}' to an int.");
                }
                string modifiedInput;

                modifiedInput = input.Replace("b", "").Replace("c", "").Replace("d", "").Replace("e", "").Replace("f", "").Replace("g", "").Replace("h", "").Replace("i", "").Replace("j", "");
                modifiedInput = modifiedInput.Replace("k", "").Replace("l", "").Replace("m", "").Replace("n", "").Replace("o", "").Replace("p", "").Replace("q", "").Replace("r", "").Replace("s", "");
                modifiedInput = modifiedInput.Replace("t", "").Replace("u", "").Replace("v", "").Replace("w", "").Replace("x", "").Replace("y", "").Replace("z", "");
                modifiedInput = modifiedInput.Replace("B", "").Replace("C", "").Replace("D", "").Replace("E", "").Replace("F", "").Replace("G", "").Replace("H", "").Replace("I", "").Replace("J", "");
                modifiedInput = modifiedInput.Replace("K", "").Replace("L", "").Replace("M", "").Replace("N", "").Replace("O", "").Replace("P", "").Replace("Q", "").Replace("R", "").Replace("S", "");
                modifiedInput = modifiedInput.Replace("T", "").Replace("U", "").Replace("V", "").Replace("W", "").Replace("X", "").Replace("Y", "").Replace("Z", "").Replace("A", "");

                //string result = Regex.Replace(input, "[^0-9.]", "");

                modifiedInput = modifiedInput.Replace("하나", "1").Replace("둘", "2").Replace("셋", "3").Replace("넷", "4").Replace("다섯", "5").Replace("여섯", "6");
                modifiedInput = modifiedInput.Replace("일곱", "7").Replace("여덟", "8").Replace("아홉", "9").Replace("열", "10");



                Console.WriteLine(modifiedInput);
                //string input_read = Console.ReadLine();
                string input_read = modifiedInput.Substring(0);
                // 문자열에 문제가 있습니다. 다시 입력해주세요
                // 이 구문 만들기 for문으로 

                for (int index = 0; index < 1; ++index)
                {
                    //modifiedInput = modifiedInput.ToString(); 
                    if (Int32.TryParse(modifiedInput, out int modifiedInputs))
                    {
                        modifiedInputs.ToString();
                        modifiedInput = modifiedInputs.ToString();
                    }
                    else if (modifiedInput.Contains("a"))
                    {
                        modifiedInput = modifiedInput.Replace("a", "-595959");
                        try
                        {
                            //long.Parse(modifiedInput); 
                            modifiedInput.ToString();
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine("문자열 잘못됨");
                                                 
                        }
                    }
                    else
                    {
                        Console.WriteLine("문자열 잘못됨");
                        continue;
                    }
                    


                }
                //input_read = input_read.Replace("a", "595959");
                modifiedInput = modifiedInput.Replace("-595959", "a");
                if (input_read == modifiedInput)
                {
                    string modifiedInput_fianl;


                    modifiedInput_fianl = modifiedInput.Replace("a", "");

                    if (Int32.TryParse(modifiedInput_fianl, out int modifiedInputs))
                    {

                        int sum = 0;
                        for (int index = 0; index < modifiedInput.Length; ++index)
                        {
                            bool aFlag = false;
                            int currentNum = 0;
                            try
                            {
                                currentNum = int.Parse(modifiedInput[index].ToString());
                            }
                            catch (Exception ex)
                            {
                                continue;
                            }

                            if (index < modifiedInput.Length - 2 && modifiedInput[index + 1] == 'a')//뒤에 a 
                            {
                                aFlag = true;
                            }

                            if (aFlag == true)
                            {
                                sum += currentNum * 2;//a있을때
                                ++index; //a다음꺼 오도록
                                Console.Write(currentNum * 2);
                            }
                            else
                            {
                                sum += currentNum; //char는 string으로 변환
                                Console.Write(currentNum);
                            }

                            if (index < modifiedInput.Length - 1)
                            {

                                Console.Write("+");
                            }
                        }
                        Console.WriteLine($" = {sum}");
                    }
                    else
                        Console.WriteLine("문자열에 문제가 있습니다.");
                        continue;
                }
                else 
                {
                    continue;

                } // 여기에 및에 식 다 넣어보기
                //Int32.TryParse(modifiedInput, out int modifiedInputs_2);
                // modifiedInput의 저장된 것이랑 12a3409같으면 밑으로 다르면 위로

                /*
                if (input.Contains("a"))
                {
                    input = input.Replace("a", "*2");

                    //Console.WriteLine(input);
                    int f;
                    f = input.IndexOf("*2");
                    //Console.WriteLine(f);
                    string v = f.ToString();
                    string g = input.ToString().Substring(f - 1, f);
                    //Console.WriteLine(g);

                    string[] result = g.Split('*');

                    int sum_1 = 0;

                    for (int j = 0; j < result.Length; j++)
                    {
                        sum_1 = Int32.Parse(result[0]) * 2;
                    }
                    //Console.WriteLine(sum_1);//이거 소문자a를 앞자리와 곱한 것
                    string v1 = g.ToString();
                    string v2 = sum_1.ToString();

                    input = input.Replace(v1, v2);                                       
                }
                */


                // *2 하는 로직을 마지막에 더할때 같이 넣어주기 //

                //string[] inputs = input.Split('');
                //Console.WriteLine(input);
                //int.Parse(input);
                
                /*
                for (int h = 0; h < input.Length; h++)
                {
                    bool flag = false;
                    if (h == input.Length - 1) //마지막 문자
                    {
                        Console.Write(input[h]);
                    }
                    else
                    {
                        if (input[h + 1] == 'a')//뒤에 a 
                        {
                            flag = true;
                        }
                        Console.Write(input[h] + "+");

                    }

                    if (flag == true)
                    {
                        sum_2 += Int32.Parse(input[h].ToString()) * 2;//a있을때
                        h++; //a다음꺼 오도록
                    }
                    else
                    {
                        sum_2 += Int32.Parse(input[h].ToString()); //char는 string으로 변환
                    }
                }
                Console.WriteLine($" = {sum_2}");
                */


                
                
                
            }
            
        }
    }
}
