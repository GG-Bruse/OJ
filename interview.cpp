#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <unordered_map>
#include <queue>
using namespace std;






/*
����һ�������������ظö�������������Ľ�����������ң�һ��һ��ر�����
�����Ķ�������{3,9,20,#,#,15,7}, ���ؽ��[[3],[9,20],[15,7]]
*/
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};
class Solution 
{
public:
    vector<vector<int> > levelOrder(TreeNode* root)
    {
        if (root == nullptr) return {};

        vector<vector<int>> result;
        queue<TreeNode*> qe;
        qe.push(root);

        while (!qe.empty())
        {
            vector<int> layer;
            int size = qe.size();
            for (int i = 0; i < size; ++i)
            {
                TreeNode* node = qe.front();
                layer.push_back(node->val);
                qe.pop();
                if (node->left != nullptr) qe.push(node->left);
                if (node->right != nullptr) qe.push(node->right);
            }
            result.push_back(layer);
        }
        return result;
    }
};