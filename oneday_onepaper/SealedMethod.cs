using System;

namespace SealedMethod
{
    class Base
    {
        public virtual void SealMe()
        {

        }
    }
    class Derived : Base
    {
        public sealed override void SealMe()
        {
        }
    }
    class WantToOverride : Derived
    {
        public override void SealMe() // 본인 되어 override  할 수 없어서 에러가 난다.
        {

        }
    }

    class Program
    {
        static void Main(string[] args)
        {

        }
    }
}
