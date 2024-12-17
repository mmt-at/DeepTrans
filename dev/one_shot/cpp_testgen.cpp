/*Example for cpp function unittest generation*/
// 1.Copy source code here
#include <iostream>
int addOne(int x) {
    int m = 1;
    while (x & m) {
        x = x ^ m;
        m <<= 1;
    }
    x = x ^ m;
    return x;
}
// 2.Generate Test function
void test_addOne() {
    // <1>. Synthesize input cases
    int test_cases[] = {0, 1, 2, 3, 4, 5, 10, 15, 31, 63, 127, 255, 1023};
    int num_cases = sizeof(test_cases) / sizeof(test_cases[0]);

    // <2>. Call the tested function with the synthesized input, record the output
    for (int i = 0; i < num_cases; ++i) {
        int x = test_cases[i];
        int result = addOne(x);

        // <3>. Print the output, format: function_name(synthesized_input) = run_output
        std::cout << "addOne(" << x << ") = " << result << std::endl;
    }
}
// 3.Run the test function
int main() {
    test_addOne();
    return 0;
}