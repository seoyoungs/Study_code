using System;
using static System.Console;

namespace StringModify
{
    class MainApp
    {
        static void Main(string[] args)
        {

            WriteLine("ToLower() : '{0}'", "ABC".ToLower()); // 대문자를 소문자로 바꾼다.
            WriteLine("ToUpper() : '{0}'", "abc".ToUpper()); // 소문자를 대문자로 바꾼다.

            WriteLine("Insert() : '{0}'", "Happy Friday!".Insert(5, "Sunny")); // 지정된 문자열에 지정된 문자 반환
            WriteLine("Remove() : '{0}'", "I don't Love You".Remove(2, 6)); // 특정 문자수 삭제

            WriteLine("Trim() : '{0}'", " No Spaces ".Trim()); // 앞뒤있는 공백 삭제한 문자열 반환
            WriteLine("TrimStart() : '{0}'", " No Spaces".TrimStart()); // 앞에 공백 삭제한 문자열 반환
            WriteLine("TrimEnd() : '{0}'", "No Spaces ".TrimEnd()); // 뒤에 있는 문자열 삭제한 문자열 반환

        }
    }
}
