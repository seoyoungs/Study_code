using System;

namespace Char
{
    class Program
    {
        static void Main(string[] args)
        {
            char a= '안';
            char b= '녕';
            char c= '하';
            char d= '세';
            char e= '요';

            Console.Write(a); // Console.Write는 데이터 출력 후 줄을 바꾸지 않는다.
            Console.Write(b);
            Console.Write(c);
            Console.Write(d);
            Console.Write(e);
            Console.WriteLine(); //WriteLine 데이터 출력 후 줄을 바꾼다.
        }
    }
}
