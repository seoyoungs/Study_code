#include <string>
#include <vector>
#include <iostream>

using namespace std;

vector<int> solution(int n, long long left, long long right) {
    vector<int> answer;
    int firstindex = 0; // 2차원 배열의 행
    int lastindex = 0; // 2차원 배열의 열
    
    // 행열의 인덱스 값이 그자리의 값이다.
    //평탄화 했을 시 각 행렬은 i/n, i%n으로 나타낼 수 있음.
    for(long long i = left; i <= right; i++) //left부터 right까지 반복문 진행
    {
        firstindex = i / n;
        lastindex = i % n;
        
        // 여기 코드 줄여보기 (삼항 연산자)
        answer.push_back(firstindex > lastindex ? firstindex + 1 : lastindex + 1); 
        
        //if(firstindex >= lastindex)//2차원 배열의 행의 값이 열+1보다 크거나 같으면 answer에 추가
        //{
            //lastindex += firstindex - lastindex;
        //}
        
        //answer.push_back(lastindex + 1); // lastindex+1
    }
    
    return answer;
}    
