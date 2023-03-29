public class task5qus1 {
    public static void main(String[] args) {
        int sum = 0;

        // Using a for loop
        for (int i = 1; i <= 10; i += 2) {
            sum += i;
        }
        System.out.println("Sum of odd numbers from 1 to 10: " + sum);

        // Using a while loop
        sum = 0;
        int i = 1;
        while (i <= 10) {
            if (i % 2 != 0) {
                sum += i;
            }
            i++;
        }
        System.out.println("Sum of odd numbers from 1 to 10: " + sum);

        // Using a do-while loop
        sum = 0;
        i = 1;
        do {
            if (i % 2 != 0) {
                sum += i;
            }
            i++;
        } while (i <= 10);
        System.out.println("Sum of odd numbers from 1 to 10: " + sum);
    }
}
