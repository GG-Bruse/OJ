/*
����֮��
https://leetcode.cn/problems/two-sum/description/?envType=study-plan-v2&envId=top-100-liked
����һ����������nums��һ������Ŀ��ֵtarget,�����ڸ��������ҳ���ΪĿ��ֵtarget������������
���������ǵ������±�
����Լ���ÿ������ֻ���Ӧһ����,�����㲻��ʹ��������ͬ��Ԫ��
����԰�����˳�򷵻ش�
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
����г���
https://leetcode.cn/problems/longest-consecutive-sequence/description/?envType=study-plan-v2&envId=top-100-liked
����һ��δ������������� nums ���ҳ���������������У���Ҫ������Ԫ����ԭ�������������ĳ���
������Ʋ�ʵ��ʱ�临�Ӷ�Ϊ O(n) ���㷨���������

���룺nums = [100,4,200,1,3,2]
�����4
���ͣ���������������� [1, 2, 3, 4]�����ĳ���Ϊ 4
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
�ƶ���
https://leetcode.cn/problems/move-zeroes/description/?envType=study-plan-v2&envId=top-100-liked
����һ������ nums����дһ������������ 0 �ƶ��������ĩβ��ͬʱ���ַ���Ԫ�ص����˳��
��ע�⣬�����ڲ���������������ԭ�ض�������в���
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
ʢˮ��������
https://leetcode.cn/problems/container-with-most-water/description/?envType=study-plan-v2&envId=top-100-liked
����һ������Ϊ n ���������� height ���� n �����ߣ��� i ���ߵ������˵��� (i, 0) �� (i, height[i])
�ҳ����е������ߣ�ʹ�������� x �Ṳͬ���ɵ�����������������ˮ
�����������Դ�������ˮ��
˵�����㲻����б����
*/
#include <iostream>
#include <vector>
using namespace std;
class Solution
{
public:
    // ÿ���ƶ�����Ϊȡ�����ˮ, ���̰岻��, ȡ��ˮ��Զ�������һ�θ���
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
����֮��
https://leetcode.cn/problems/3sum/description/?envType=study-plan-v2&envId=top-100-liked
����һ���������� nums ���ж��Ƿ������Ԫ�� [nums[i], nums[j], nums[k]] ���� i != j��i != k �� j != k
ͬʱ������ nums[i] + nums[j] + nums[k] == 0
���㷵�����к�Ϊ 0 �Ҳ��ظ�����Ԫ�顣
ע�⣺���в����԰����ظ�����Ԫ��
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
        sort(nums.begin(), nums.end()); // ���ڲ����ظ�, ���������
        vector<vector<int>> result;
        for (size_t i = 0; i < size - 2; ++i)
        {
            if (i > 0 && nums[i] == nums[i - 1]) continue;//������һ��ö�ٵ�������ͬ
            int target = 0 - nums[i];
            int k = size - 1;
            for (int j = i + 1; j < size - 1; ++j)
            {
                if (j > i + 1 && nums[j] == nums[j - 1]) continue;//������һ��ö�ٵ�������ͬ
                while (j < k && nums[j] + nums[k] > target) --k;// ˫ָ�����, <��=��ֹͣ
                // nums[j] < ����whileö�ٹ���nums[k], ��nums[j]����, nums[k]���С
                // ����j == k��, ˵���������е�k, ����nums[j] + nums[k] > target
                // j�������ֻ���ܸ���, ���Ҳ������и�С��nums[k]��
                // ��ʱ��������j��ѭ�� 
                if (j == k) break;
                if (nums[j] + nums[k] == target)
                    result.push_back({ nums[i], nums[j], nums[k] });
            }
        }
        return result;
    }
};



/*
���ظ��ַ�����Ӵ�
https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/?envType=study-plan-v2&envId=top-100-liked
����һ���ַ���s �������ҳ����в������ظ��ַ�����Ӵ��ĳ���
����: s = "abcabcbb"
���: 3
����: ��Ϊ���ظ��ַ�����Ӵ��� "abc"�������䳤��Ϊ 3
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
�ҵ��ַ�����������ĸ��λ��
https://leetcode.cn/problems/find-all-anagrams-in-a-string/description/?envType=study-plan-v2&envId=top-100-liked
���������ַ��� s �� p���ҵ� s ������ p ����λ�ʵ��Ӵ���������Щ�Ӵ�����ʼ�����������Ǵ������˳��
����: s = "cbaebabacd", p = "abc"
���: [0,6]
����:
��ʼ�������� 0 ���Ӵ��� "cba", ���� "abc" ����λ��
��ʼ�������� 6 ���Ӵ��� "bac", ���� "abc" ����λ��
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
// ��������
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
            // ��󻬶�һ��
            --sHash[s[i] - 'a'];
            ++sHash[s[i + pSize] - 'a'];
            if (sHash == pHash) result.push_back(i + 1);
        }
        return result;
    }
};



/*
��Ϊ K ��������
https://leetcode.cn/problems/subarray-sum-equals-k/description/?envType=study-plan-v2&envId=top-100-liked
����һ���������� nums ��һ������ k ������ͳ�Ʋ����� �������к�Ϊ k ��������ĸ���
��������������Ԫ�ص������ǿ�����
���룺nums = [1,1,1], k = 2
�����2
*/
#include <iostream>
#include <vector>
#include <unordered_map>
using namespace std;
class Solution
{
public:
    // pre[i]Ϊnums[0...i]��������֮��, pre[i] = pre[i - 1] + nums[i]
    //[i,j]��Ϊk������, ��prev[j] - prev[i - 1] == k, ����Ϊpre[i - 1] = prev[j] - k
    int subarraySum(vector<int>& nums, int k)
    {
        // ��¼prev[i]���ڸ���
        unordered_map<int, int> hashMap;
        hashMap[0] = 1;

        int result = 0;
        int pre = 0;
        for (int i = 0; i < nums.size(); ++i)
        {
            pre += nums[i];// �������ӵ�prev[j]
            // �Ƿ����prev[j] - k, ��pre[i - 1]
            if (hashMap.find(pre - k) != hashMap.end())
                result += hashMap[pre - k];
            ++hashMap[pre];
        }
        return result;
    }
};