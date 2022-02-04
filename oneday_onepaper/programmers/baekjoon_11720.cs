using System;
using System.Diagnostics.CodeAnalysis;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.IO.Ports;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace baekjoon_11720
{
    class Program
    {
        static void Main(string[] args)
        {

            //int num = int.Parse(Console.ReadLine());

            //string sentence = Console.ReadLine();

            //int sum = 0;

            //for (int i = 0; i < num; i++)
            //{
            //    sum += int.Parse(sentence[i].ToString());
            //}
            int[] nums = { 2, 7, 11, 15 };
            int target = 9;
            int[] sum = new int[2];
            for (int i = 0; i < nums.Length - 1; i++)
            {
                int s = target - nums[i];
                for (int j = i + 1; j > nums.Length - 1; j++)
                {
                    if (nums[i] + nums[j] == target)
                    {
                        sum[0] = i;
                        sum[1] = j;

                    }
                }

            }
            Console.WriteLine($"{sum}");
        }

        public int[] TwoSum(int[] nums, int target)
        {
            int[] sum = new int[2];
            for (int i = 0; i < nums.Length - 1; i++)
            {
                int s = target - nums[i];
                for (int j = i + 1; j < nums.Length - 1; j++)
                {
                    if (nums[j] == s)
                    {
                        sum[0] = i;
                        sum[1] = j;

                    }
                }

            }
            return sum;

        }

    }
}
/*
    class baekjoon
    {
        static void baekjoon_9498(string[] args)
        {
            // 첫째줄
            int score = int.Parse(Console.ReadLine());
            if (score > 100 || score < 0) return;


            //둘째줄 출력 : 시험점수가 출력된다
            if (score >= 90 && score <= 100)
            {
                Console.WriteLine("A");
            }
            else if (score >= 80 && score <= 89)
            {
                Console.WriteLine("B");
            }
            else if (score >= 70 && score <= 79)
            {
                Console.WriteLine("C");
            }
            else if (score >= 60 && score <= 69)
            {
                Console.WriteLine("D");
            }
            else
                Console.WriteLine("F");
            

        }

        public int[] TwoSum(int[] nums, int target)
        {
            int[] sum = new int[2];
            for (int i = 0; i < nums.Length - 1; i++)
            {
                int s = target - nums[i];
                for (int j = i + 1; j > nums.Length - 1; j++)
                {
                    if (nums[j] == s)
                    {
                        sum[0] = i;
                        sum[1] = j;

                    }
                }

            }
            return sum;

        }
    }
}


 * 문제
N개의 숫자가 공백 없이 쓰여있다. 이 숫자를 모두 합해서 출력하는 프로그램을 작성하시오.

입력
첫째 줄에 숫자의 개수 N (1 ≤ N ≤ 100)이 주어진다. 둘째 줄에 숫자 N개가 공백없이 주어진다.

출력
입력으로 주어진 숫자 N개의 합을 출력한다.

예제 입력 1 
1
1
예제 출력 1 
1

예제 입력 2 
5
54321
예제 출력 2 
15


백준 9498번
문제
시험 점수를 입력받아 90 ~ 100점은 A, 80 ~ 89점은 B, 70 ~ 79점은 C, 60 ~ 69점은 D, 나머지 점수는 F를 출력하는 프로그램을 작성하시오.

입력
첫째 줄에 시험 점수가 주어진다. 시험 점수는 0보다 크거나 같고, 100보다 작거나 같은 정수이다.

출력
시험 성적을 출력한다.

예제 입력 1 
100
예제 출력 1 
A
 */
