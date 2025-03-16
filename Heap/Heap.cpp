#include <iostream>
#include <vector>

using namespace std;

//小堆为例

void AdjustDown(vector<int>& numbers, int current, int size)
{
    int child = current * 2 + 1;
    while(child < size)
    {
        if(child + 1 < size && numbers[child + 1] < numbers[child]) ++child;
        if(numbers[current] > numbers[child]) {
            swap(numbers[current], numbers[child]);
            current = child;
            child = current * 2 + 1;
        }
        else break;
    }
}

void AdjustUp(vector<int>& numbers, int current)
{
    int parent = (current - 1) / 2;
    while(numbers[current] < numbers[parent] && current != 0) {
        swap(numbers[current], numbers[parent]);
        current = parent;
        parent = (current - 1) / 2;
    }
}

//小堆，排降序
void HeapSort(vector<int>& numbers)
{
    int size = numbers.size();
    for(int i = size/2 - 1; i >= 0; --i)
        AdjustDown(numbers, i, size);
    for(int i = size - 1; i > 0; --i) {
        swap(numbers[0], numbers[i]);
        AdjustDown(numbers, 0, i);
    }
}


vector<int> TopK(vector<int>& numbers, int k)
{
    vector<int> topk(k);

    for(int i = 0; i < k; ++i)
        topk[i] = numbers[i];
    for(int i = k / 2 - 1; i >= 0; --i)
        AdjustDown(topk, i, k);

    for(int i = k; i < numbers.size(); ++i)
    {
        if(numbers[i] > topk[0]) {
            swap(numbers[i], topk[0]);
            AdjustDown(topk, 0, k);
        }
    }
    return topk;
}


int main()
{
    vector<int> numbers = {1,3,5,7,9,2,4,6,8,10};
    // HeapSort(numbers);
    numbers = TopK(numbers, 3);
    for(auto& it : numbers) cout << it << " ";
    cout << endl;
    return 0;
}