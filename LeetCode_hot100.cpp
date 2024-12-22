/*
两数之和
https://leetcode.cn/problems/two-sum/description/?envType=study-plan-v2&envId=top-100-liked
给定一个整数数组nums和一个整数目标值target,请你在该数组中找出和为目标值target的那两个整数
并返回它们的数组下标
你可以假设每种输入只会对应一个答案,并且你不能使用两次相同的元素
你可以按任意顺序返回答案
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
字母异位词分组
https://leetcode.cn/problems/group-anagrams/description/?envType=study-plan-v2&envId=top-100-liked
给你一个字符串数组，请你将字母异位词组合在一起。可以按任意顺序返回结果列表
字母异位词是由重新排列源单词的所有字母得到的一个新单词
输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
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

