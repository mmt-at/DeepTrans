/*Example for c function unittest generation*/
// 1.Copy source code here, add include header of the function to be tested
#include <stdio.h>
#include "addOne.h"

// 2.Generate Test function
void test_addOne() {
    // <1>. Given: Set up any necessary conditions or inputs.
    int test_cases[] = {0, 1, 2, 3, 4, 5, 10, 15, 31, 63, 127, 255, 1023};
    int num_cases = sizeof(test_cases) / sizeof(test_cases[0]);

    // <2>. When: Perform the action (calling the function).
    for (int i = 0; i < num_cases; ++i) {
        int x = test_cases[i];
        int result = addOne(x);

        // <3>. Then: Verify/print the output or outcome.
        printf("addOne(%d) = %d\n", x, result);
    }
}

// 3.Run the test function
int main() {
    test_addOne();
    return 0;
}