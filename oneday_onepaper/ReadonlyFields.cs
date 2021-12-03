using System;

namespace ReadonlyFields
{

    class Configuration
    {
        private readonly int min; // readonly를 이용해 읽기 전용 필드를 선언
        private readonly int max;

        public Configuration(int v1, int v2)
        {
            min = v1;
            max = v2; // 읽기 전용 필드는 생성자 안에서만 초기화가 가능
        }

        public void ChangeMax(int newMax)
        {
            max = newMax; // 생성자가 아닌 곳에서 값 수정하면 에러
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            Configuration c = new Configuration(100, 10);
        }
    }
}
