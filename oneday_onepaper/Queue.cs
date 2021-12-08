using System;
using System.Collections.Generic;

namespace Queue
{
    class Program
    {
        static void Main(string[] args)
        {
            Queue<int> que = new Queue<int>(); // queue를 저장할 정수 배열 생성
            //Queue que = new Queue();
            
            que.Enqueue(1);
            que.Enqueue(2);
            que.Enqueue(3);
            que.Enqueue(4);
            que.Enqueue(5);

            Console.WriteLine("que.Count: {0}", que.Count); //queue에 저장된 개수
            Console.WriteLine();

            while (que.Count > 0)
                Console.WriteLine(que.Dequeue());
            Console.WriteLine();

            Console.WriteLine("que.Count: {0}", que.Count);
        }
    }
}
