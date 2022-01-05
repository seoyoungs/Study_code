using System;

namespace Overriding
{
    class ArmorSuite
    {
        public virtual void Initialize()
        {
            Console.WriteLine("Aemored");
        }
    }
    
    class IronMan : ArmorSuite
    {
        public override void Initialize()
        {
            base.Initialize(); //이렇게 불러와야한다.
            Console.WriteLine("Repulsor Rays Armed");
        }
    }
    
    class Inchon : ArmorSuite
    {
        public override void Initialize()
        {
            base.Initialize();
            Console.WriteLine("test를 해보자구나");
        }
    }

    class WarMachine : ArmorSuite
    {
        public override void Initialize()
        {
            base.Initialize();
            Console.WriteLine("Double-Barrel Cannons Armed");
            Console.WriteLine("Micro-Rocket Launcher Armed");
        }
    }

    class Program
    { 
        static void Main(string[] args)
        {
            Console.WriteLine("Creating ArmorSuite...");
            ArmorSuite armorSuite = new ArmorSuite();
            armorSuite.Initialize();

            Console.WriteLine("\nCreating IronMan...");
            ArmorSuite ironman = new IronMan();
            ironman.Initialize();

            Console.WriteLine("\nCreating WarMachine...");
            ArmorSuite warmachine = new WarMachine();
            warmachine.Initialize();

            Console.WriteLine("\nCreating WarMachine...");
            ArmorSuite inchon = new Inchon();
            inchon.Initialize();
        }
    }
}
