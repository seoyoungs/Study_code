using System;
using MyExtension;

namespace MyExtension
{
    public static class IntegerExtension
    {
        public static int Square(this int myint)
        {
            return myint * myint;
        }

        public static int Power(this int myint, int exponent)
        {
            int result = myint;
            for (int i = 1; i < exponent; i++)
                result = result * myint;

            return result;
        }
    }
}
namespace ExtensionMethod 
{ 
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine($"3^2 : {3.Square()}");
            Console.WriteLine($"3^4 : {3.Power(4)}");
            Console.WriteLine($"2^10 : {2.Power(10)}");
        }
    }
    
}
