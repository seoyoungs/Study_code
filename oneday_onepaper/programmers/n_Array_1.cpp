#include <string>
#include <vector>
#include <iostream>

using namespace std;

vector<int> solution(int n, long long left, long long right) {
    vector<int> answer;
    long long temp = 0; // 담을 수 있는 그릇 넓히기 --> left, right에 맞추기

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (temp >= right + 1) {
                return answer; // index+1 이면 그대로 리턴
            }
            else if (left <= temp) // left부터 right 값까지 모두 찾음 (temp가 커야 실행가능)
            {
                if (i <= j) {
                    answer.push_back(j + 1); // 4번 문항 참고
                }

                else { 
                    answer.push_back(i + 1);
                }
            }
            temp++;
        }
    }

    return answer;
}
