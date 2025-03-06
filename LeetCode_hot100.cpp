/*
两数之和
https://leetcode.cn/problems/two-sum/description/?envType=study-plan-v2&envId=top-100-liked
给定一个整数数组nums和一个整数目标值target,请你在该数组中找出和为目标值target的那两个整数
并返回它们的数组下标
你可以假设每种输入只会对应一个答案,并且你不能使用两次相同的元素
你可以按任意顺序返回答案
*/
#include <iostream>
#include <vector>
#include <unordered_map>
using namespace std;
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target)
    {
        unordered_map<int, int> hashMap;
        for (int i = 0; i < nums.size(); ++i)
        {
            unordered_map<int, int>::iterator it = hashMap.find(target - nums[i]);
            if (it != hashMap.end()) return { i, it->second };
            else hashMap[nums[i]] = i;
        }
        return {};
    }
};



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
最长序列长度
https://leetcode.cn/problems/longest-consecutive-sequence/description/?envType=study-plan-v2&envId=top-100-liked
给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度
请你设计并实现时间复杂度为 O(n) 的算法解决此问题

输入：nums = [100,4,200,1,3,2]
输出：4
解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4
*/
#include <iostream>
#include <unordered_set>
using namespace std;
class Solution
{
public:
    int longestConsecutive(vector<int>& nums)
    {
        unordered_set<int> hashSet;
        for (size_t i = 0; i < nums.size(); ++i)
            hashSet.insert(nums[i]);

        int longest = 0;
        for (auto& number : hashSet)
        {
            if (hashSet.count(number - 1)) continue;
            else
            {
                int currentNumber = number;
                int length = 1;
                while (hashSet.count(currentNumber + 1)) {
                    currentNumber += 1;
                    length += 1;
                }
                longest = max(longest, length);
                if (hashSet.size() == longest) break;
            }
        }
        return longest;
    }
};



/*
移动零
https://leetcode.cn/problems/move-zeroes/description/?envType=study-plan-v2&envId=top-100-liked
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序
请注意，必须在不复制数组的情况下原地对数组进行操作
*/
#include <iostream>
#include <vector>
using namespace std;
class Solution
{
public:
    void moveZeroes(vector<int>& nums)
    {
        int start = 0, end = 0;
        while (end < nums.size())
        {
            if (0 != nums[end]) {
                swap(nums[start], nums[end]);
                ++start;
            }
            ++end;
        }
    }
};



/*
盛水最多的容器
https://leetcode.cn/problems/container-with-most-water/description/?envType=study-plan-v2&envId=top-100-liked
给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i])
找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水
返回容器可以储存的最大水量
说明：你不能倾斜容器
*/
#include <iostream>
#include <vector>
using namespace std;
class Solution
{
public:
    // 每次移动都是为取更多的水, 若短板不动, 取的水永远不会比上一次更多
    int maxArea(vector<int>& height)
    {
        int left = 0, right = height.size() - 1;
        int result = 0;
        while (left < right)
        {
            int area = min(height[left], height[right]) * (right - left);
            result = max(area, result);
            if (height[left] <= height[right]) ++left;
            else --right;
        }
        return result;
    }
};



/*
三数之和
https://leetcode.cn/problems/3sum/description/?envType=study-plan-v2&envId=top-100-liked
给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k
同时还满足 nums[i] + nums[j] + nums[k] == 0
请你返回所有和为 0 且不重复的三元组。
注意：答案中不可以包含重复的三元组
*/
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
class Solution
{
public:
    vector<vector<int>> threeSum(vector<int>& nums)
    {
        int size = nums.size();
        sort(nums.begin(), nums.end()); // 由于不可重复, 需进行排序
        vector<vector<int>> result;
        for (size_t i = 0; i < size - 2; ++i)
        {
            if (i > 0 && nums[i] == nums[i - 1]) continue;//需与上一次枚举的数不相同
            int target = 0 - nums[i];
            int k = size - 1;
            for (int j = i + 1; j < size - 1; ++j)
            {
                if (j > i + 1 && nums[j] == nums[j - 1]) continue;//需与上一次枚举的数不相同
                while (j < k && nums[j] + nums[k] > target) --k;// 双指针查找, <、=都停止
                // nums[j] < 所有while枚举过的nums[k], 若nums[j]增大, nums[k]需减小
                // 但当j == k是, 说明找了所有的k, 都是nums[j] + nums[k] > target
                // j继续向后只可能更大, 并且不可能有更小的nums[k]了
                // 此时结束本次j的循环 
                if (j == k) break;
                if (nums[j] + nums[k] == target)
                    result.push_back({ nums[i], nums[j], nums[k] });
            }
        }
        return result;
    }
};



/*
无重复字符的最长子串
https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/?envType=study-plan-v2&envId=top-100-liked
给定一个字符串s ，请你找出其中不含有重复字符的最长子串的长度
输入: s = "abcabcbb"
输出: 3
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3
*/
#include <iostream>
#include <string>
#include <unordered_map>
using namespace std;
class Solution
{
public:
    int lengthOfLongestSubstring(string s)
    {
        int size = s.size();
        if (0 == size) return 0;

        int maxLength = 1;
        int prev = 0, last = 0;
        unordered_map<char, int> hashCount;
        while (prev < size)
        {
            while (last < size && hashCount[s[last]] == 0) {
                hashCount[s[last]]++;
                ++last;
            }
            maxLength = max(maxLength, last - prev);
            hashCount[s[prev]]--;
            ++prev;
        }
        return maxLength;
    }
};



/*
找到字符串中所有字母异位词
https://leetcode.cn/problems/find-all-anagrams-in-a-string/description/?envType=study-plan-v2&envId=top-100-liked
给定两个字符串 s 和 p，找到 s 中所有 p 的异位词的子串，返回这些子串的起始索引。不考虑答案输出的顺序
输入: s = "cbaebabacd", p = "abc"
输出: [0,6]
解释:
起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词
起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词
*/
// error : timeout
class Solution
{
public:
    vector<int> findAnagrams(string s, string p)
    {
        vector<int> result;
        unordered_map<char, int> hashExist;
        for (char& ch : p) hashExist[ch]++;

        int size = s.size();
        int prev = 0, last = 0;
        unordered_map<char, int> hashCurrent;
        while (prev < size)
        {
            while (last < size && hashCurrent[s[last]] < hashExist[s[last]]) {
                hashCurrent[s[last]]++;
                ++last;
            }
            bool isHeterotopicWords = true;
            for (auto& ch : p) {
                if (hashCurrent[ch] != hashExist[ch]) {
                    isHeterotopicWords = false;
                    break;
                }
            }
            if (isHeterotopicWords) result.push_back(prev);
            ++prev;
            last = prev;
            hashCurrent.clear();
        }
        return result;
    }
};
// 滑动窗口
#include <iostream>
#include <vector>
#include <string>
using namespace std;
class Solution
{
public:
    vector<int> findAnagrams(string s, string p)
    {
        int sSize = s.size(), pSize = p.size();
        if (sSize < pSize) return vector<int>();

        vector<int> result;
        vector<int> sHash(26);
        vector<int> pHash(26);
        for (int i = 0; i < pSize; ++i) {
            ++sHash[s[i] - 'a'];
            ++pHash[p[i] - 'a'];
        }

        if (sHash == pHash) result.push_back(0);

        for (int i = 0; i < sSize - pSize; ++i) {
            // 向后滑动一次
            --sHash[s[i] - 'a'];
            ++sHash[s[i + pSize] - 'a'];
            if (sHash == pHash) result.push_back(i + 1);
        }
        return result;
    }
};



/*
和为 K 的子数组
https://leetcode.cn/problems/subarray-sum-equals-k/description/?envType=study-plan-v2&envId=top-100-liked
给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的子数组的个数
子数组是数组中元素的连续非空序列
输入：nums = [1,1,1], k = 2
输出：2
*/
#include <iostream>
#include <vector>
#include <unordered_map>
using namespace std;
class Solution
{
public:
    // pre[i]为nums[0...i]的所有数之和, pre[i] = pre[i - 1] + nums[i]
    //[i,j]和为k的数组, 即prev[j] - prev[i - 1] == k, 移项为pre[i - 1] = prev[j] - k
    int subarraySum(vector<int>& nums, int k)
    {
        // 记录prev[i]存在个数
        unordered_map<int, int> hashMap;
        hashMap[0] = 1;

        int result = 0;
        int pre = 0;
        for (int i = 0; i < nums.size(); ++i)
        {
            pre += nums[i];// 不断增加的prev[j]
            // 是否存在prev[j] - k, 即pre[i - 1]
            if (hashMap.find(pre - k) != hashMap.end())
                result += hashMap[pre - k];
            ++hashMap[pre];
        }
        return result;
    }
};



/*
滑动窗口最大值
https://leetcode.cn/problems/sliding-window-maximum/description/?envType=study-plan-v2&envId=top-100-liked
给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧
你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位
返回 滑动窗口中的最大值
输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]
*/
#include <iostream>
#include <queue>
#include <vector>
using namespace std;
class Solution
{
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k)
    {
        int size = nums.size();
        priority_queue<pair<int, int>> qe;
        for (int i = 0; i < k; ++i)
            qe.emplace(nums[i], i);
        vector<int> result = { qe.top().first };

        for (int i = k; i < size; ++i) //不断移动的窗口右边界
        {
            qe.emplace(nums[i], i);
            while (qe.top().second <= i - k) qe.pop();//若堆顶元素不在滑动窗口内则删除
            result.push_back(qe.top().first);
        }
        return result;
    }
};



/*
最小覆盖子串
https://leetcode.cn/problems/minimum-window-substring/description/?envType=study-plan-v2&envId=top-100-liked
给你一个字符串 s 、一个字符串 t
返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 ""
注意:
对于 t 中重复字符，我们寻找的子字符串中该字符数量必须不少于 t 中该字符数量
如果 s 中存在这样的子串，我们保证它是唯一的答案
输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
解释：最小覆盖子串 "BANC" 包含来自字符串 t 的 'A'、'B' 和 'C'
*/
#include <iostream>
#include <string>
#include <unordered_map>
using namespace std;
class Solution
{
public:
    bool check() // 检查当前窗口是否包含t所需的所有字符
    {
        for (const auto& it : origin)
            if (count[it.first] < it.second)
                return false;
        return true;
    }

    string minWindow(string s, string t)
    {
        for (const char& ch : t) ++origin[ch];

        int left = 0, right = -1;
        int resultLeft = -1, resultRight = -1;
        int length = INT_MAX;
        int size = s.size();

        while (right < size) //右边界不断扩大
        {
            // 右边界指向的字符若存在于t中, count进行记录
            if (origin.find(s[++right]) != origin.end()) ++count[s[right]];
            // 收缩左边界, 符合要求的前提下, 越小越好
            while (check() && left <= right)
            {
                if (right - left + 1 < length) { //修改result
                    length = right - left + 1;
                    resultLeft = left;
                }
                if (origin.find(s[left]) != origin.end()) //删除count中的记录
                    --count[s[left]];
                ++left;
            }
        }
        return resultLeft == -1 ? string() : s.substr(resultLeft, length);
    }
private:
    unordered_map<char, int> origin, count;
    // origin 存放t中各个元素出现的次数
    // count 维护当前滑动窗口中各个元素出现的次数
};



/*
最大连续子数组
https://leetcode.cn/problems/maximum-subarray/description/?envType=study-plan-v2&envId=top-100-liked
给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和
子数组是数组中的一个连续部分
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
*/
class Solution
{
public:
    int maxSubArray(vector<int>& nums)
    {
        int size = nums.size();
        int result = nums[0];
        // dp[i] : 以nums[i]为结尾的数组的 连续子数组的最大和
        vector<int> dp(size);
        dp[0] = nums[0];
        for (int i = 1; i < size; ++i) {
            dp[i] = max(dp[i - 1] + nums[i], nums[i]);
            result = max(result, dp[i]);
        }
        return result;
    }
};
// 空间优化
class Solution
{
public:
    int maxSubArray(vector<int>& nums)
    {
        int size = nums.size();
        int result = nums[0];
        int prev = nums[0];
        for (int i = 1; i < size; ++i) {
            prev = max(prev + nums[i], nums[i]);
            result = max(result, prev);
        }
        return result;
    }
};



/*
合并区间
https://leetcode.cn/problems/merge-intervals/description/?envType=study-plan-v2&envId=top-100-liked
以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi]
请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间
*/
class Solution
{
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals)
    {
        // sort(intervals.begin(), intervals.end(), [](const vector<int>& val1, const vector<int>& val2){
        //     return val1[0] < val2[0];
        // });
        // 不写lambda,默认按照第一个数排列
        sort(intervals.begin(), intervals.end());
        vector<vector<int>> merged;
        for (int i = 0; i < intervals.size(); ++i)
        {
            int left = intervals[i][0], right = intervals[i][1];
            if (!merged.size() || merged.back()[1] < left)
                merged.push_back({ left, right });
            else
                merged.back()[1] = max(merged.back()[1], right);
        }
        return merged;
    }
};



/*
轮转数组
https://leetcode.cn/problems/rotate-array/description/?envType=study-plan-v2&envId=top-100-liked
给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数
输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右轮转 1 步: [7,1,2,3,4,5,6]
向右轮转 2 步: [6,7,1,2,3,4,5]
向右轮转 3 步: [5,6,7,1,2,3,4]
*/
class Solution
{
public:
    void rotate(vector<int>& nums, int k)
    {
        if (k == 0) return;
        int size = nums.size();
        vector<int> result(size);
        for (int i = 0; i < size; ++i)
            result[(i + k) % size] = nums[i];
        nums.assign(result.begin(), result.end());
    }
};



/*
除自身以外数组的乘积
https://leetcode.cn/problems/product-of-array-except-self/?envType=study-plan-v2&envId=top-100-liked
给你一个整数数组 nums，返回 数组 answer ，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积
题目数据 保证 数组 nums之中任意元素的全部前缀元素和后缀的乘积都在  32 位 整数范围内
请不要使用除法，且在 O(n) 时间复杂度内完成此题
输入: nums = [1,2,3,4]
输出: [24,12,8,6]
*/
class Solution
{
public:
    vector<int> productExceptSelf(vector<int>& nums)
    {
        int size = nums.size();
        vector<int> prevMulSum(size), lastMulSum(size);
        prevMulSum[0] = 1;
        for (int i = 1; i < size; ++i)
            prevMulSum[i] = prevMulSum[i - 1] * nums[i - 1];
        lastMulSum[size - 1] = 1;
        for (int i = size - 2; i >= 0; --i)
            lastMulSum[i] = lastMulSum[i + 1] * nums[i + 1];

        vector<int> result(size);
        for (int i = 0; i < size; ++i)
            result[i] = prevMulSum[i] * lastMulSum[i];
        return result;
    }
};





/*
缺失的第一个正数
https://leetcode.cn/problems/first-missing-positive/description/?envType=study-plan-v2&envId=top-100-liked
给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数
请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案
*/
class Solution
{
public:
    int firstMissingPositive(vector<int>& nums)
    {
        int size = nums.size();
        // 所有负数标记为size + 1
        for (int i = 0; i < size; ++i)
            if (nums[i] <= 0)
                nums[i] = size + 1;
        // 将绝对值小于size的元素映射的位置变成负数
        for (int i = 0; i < size; ++i) {
            int number = abs(nums[i]);
            if (number <= size) // 所有绝对值大于size的都不处理
                nums[number - 1] = -abs(nums[number - 1]);
        }
        // 返回第一个大于0的元素的下标
        for (int i = 0; i < size; ++i)
            if (nums[i] > 0)
                return i + 1;
        return size + 1;
    }
};
class Solution
{
public:
    int firstMissingPositive(vector<int>& nums)
    {
        int size = nums.size();
        for (int i = 0; i < size; ++i) {
            // 每一个while循环将 i 位置的上的数字放在其该出现的地方, 由于是交换, 新的nums[i]可能也需要放置, 所以采用循环
            while (nums[i] > 0 && nums[i] <= size && nums[nums[i] - 1] != nums[i])
                swap(nums[nums[i] - 1], nums[i]);
        }
        for (int i = 0; i < size; ++i)
            if (nums[i] != i + 1)
                return i + 1;
        return size + 1;
    }
};