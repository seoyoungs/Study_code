using System;

namespace ReadonlyMethod
{
    struct ACSetting
    {
        public double currentInCelsius;
        public double target;

        public readonly double GetFahrenheit()
        {
            //화씨(F) 계산 결과를 target에 저장
            target = currentInCelsius*1.8 + 32; // readonly이기 때문에 읽을 수 없다.
            return target; //target반환
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            ACSetting acs;
            acs.currentInCelsius = 25;
            acs.target = 25;

            Console.WriteLine($"{acs.GetFahrenheit()}");
            Console.WriteLine($"{acs.target}");
        }
    }
}
