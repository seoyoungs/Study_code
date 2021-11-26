using System;

namespace ShiftOperator
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Testing <<...");
            int a = 1;
            Console.WriteLine("a      :{0:D5}  (0x{0:x8})", a);
            Console.WriteLine("a << 1 :{0:D5}  (0x{0:x8})", a <<1); // 왼쪽으로 한 칸이동 2^1
            Console.WriteLine("a << 2 :{0:D5}  (0x{0:x8})", a <<2); // 왼쪽으로 두 칸이동 2^2
            Console.WriteLine("a << 5 :{0:D5}  (0x{0:x8})", a <<5); // 왼쪽으로 두 칸이동 2^5

            int b = 255;
            Console.WriteLine("b      : {0:D5}  (0x{0:x8})", b);
            Console.WriteLine("b >> 1 : {0:D5}  (0x{0:x8})", b>>1);
            Console.WriteLine("b >> 2 : {0:D5}  (0x{0:x8})", b>>2); // 오른쪽으로 이동 (값이 작아짐)
            Console.WriteLine("b >> 5 : {0:D5}  (0x{0:x8})", b>>5);

            int c = -255;
            Console.WriteLine("c      : {0:D5}  (0x{0:x8})", c);
            Console.WriteLine("c >> 1 : {0:D5}  (0x{0:x8})", c >> 1);
            Console.WriteLine("c >> 2 : {0:D5}  (0x{0:x8})", c >> 2);
            Console.WriteLine("c >> 5 : {0:D5}  (0x{0:x8})", c >> 5);
            
        }
    }
}
