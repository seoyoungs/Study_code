using static System.Console;

namespace StringSearch
{
    class MainApp
    {
        static void Main(string[] args)
        {
            string greeting = "Good Morning";

            WriteLine(greeting);
            WriteLine();

            // IndexOf()
            WriteLine("IndexOf 'Good' : {0}", greeting.IndexOf("Good")); //IndexOf: Good의 위치를 찾는다.
            WriteLine("IndexOf 'o' : {0}", greeting.IndexOf('o'));

            // LastIndexOf()
            WriteLine("LastIndexOf 'Good' ; {0}", greeting.LastIndexOf("Good")); //문자열 위치를 뒤에서 부터 찾습니다.
            WriteLine("IndexOf 'o' : {0}", greeting.LastIndexOf('o'));

            // StartsWith()
            WriteLine("LastIndexOf 'Good' ; {0}", greeting.StartsWith("Good")); // 현 문자열이 지정된 문자열로 시작하는지 평가
            WriteLine("IndexOf 'o' : {0}", greeting.StartsWith("Morning"));

            // EndsWith()
            WriteLine("LastIndexOf 'Good' ; {0}", greeting.EndsWith("Good")); // 현 문자열이 지정된 문자열로 끝나는지 평가
            WriteLine("IndexOf 'o' : {0}", greeting.EndsWith("Morning"));

            // Contains()
            WriteLine("LastIndexOf 'Good' ; {0}", greeting.Contains("Evening")); // 현 문자열이 포함하는지
            WriteLine("IndexOf 'o' : {0}", greeting.Contains("Morning"));

            // Replace()
            WriteLine("Replaced 'Morning' with 'Evening': {0}",
                greeting.Replace("Morning", "Evening")); // 다른 문자열로 대체하기

        }
    }
}
