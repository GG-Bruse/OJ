/*
����֮��
https://leetcode.cn/problems/two-sum/description/?envType=study-plan-v2&envId=top-100-liked
����һ����������nums��һ������Ŀ��ֵtarget,�����ڸ��������ҳ���ΪĿ��ֵtarget������������
���������ǵ������±�
����Լ���ÿ������ֻ���Ӧһ����,�����㲻��ʹ��������ͬ��Ԫ��
����԰�����˳�򷵻ش�
*/
//#include <iostream>
//#include <vector>
//#include <unordered_map>
//using namespace std;
//class Solution {
//public:
//    vector<int> twoSum(vector<int>& nums, int target)
//    {
//        unordered_map<int, int> hashMap;
//        for (int i = 0; i < nums.size(); ++i)
//        {
//            unordered_map<int, int>::iterator it = hashMap.find(target - nums[i]);
//            if (it != hashMap.end()) return { i, it->second };
//            else hashMap[nums[i]] = i;
//        }
//        return {};
//    }
//};



/*
��ĸ��λ�ʷ���
https://leetcode.cn/problems/group-anagrams/description/?envType=study-plan-v2&envId=top-100-liked
����һ���ַ������飬���㽫��ĸ��λ�������һ�𡣿��԰�����˳�򷵻ؽ���б�
��ĸ��λ��������������Դ���ʵ�������ĸ�õ���һ���µ���
����: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
���: [["bat"],["nat","tan"],["ate","eat","tea"]]
*/
#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
using namespace std;
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs)
    {
        unordered_map<string, vector<string>> hashMap;
        for (string& str : strs) {
            string tmp = str;
            sort(tmp.begin(), tmp.end());
            hashMap[tmp].push_back(str);
        }
        vector<vector<string>> result;
        for (auto it = hashMap.begin(); it != hashMap.end(); ++it)
            result.push_back(it->second);
        return result;
    }
};



/*

*/

