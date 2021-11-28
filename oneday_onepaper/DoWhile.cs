using System;

namespace DoWhile
{
    class Program
    {
        static void Main(string[] args)
        {
            int i = 10;

            do
            {
                Console.WriteLine("a) i :{0}", i--);
            }
            while (i > 0);

            do
            {
                //0이지만 여기서도 한차례 실행
                Console.WriteLine("b) i :{0}", i--);
            }
            while (i > 0);
        }
    }
}
