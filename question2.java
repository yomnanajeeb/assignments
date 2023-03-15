import java.util.Scanner;

public class question2 {
    public static void main(String[] args) {
        Scanner scanner =new Scanner(System.in);
        Scanner input = new Scanner(System.in);
        int num1, num2;
        char operator;
        double result;

        System.out.println("Enter first number: ");
        num1 = input.nextInt();

        System.out.println("Enter second number: ");
        num2 = input.nextInt();

        System.out.println("Enter an operator (+, -, *, /): ");
        operator = input.next().charAt(0);

        if (operator == '+') {
            result = num1 + num2;
        } else if (operator == '-') {
            result = num1 - num2;
        } else if (operator == '*') {
            result = num1 * num2;
        } else if (operator == '/') {
            result = num1 / num2;
        } else {
            System.out.println("Invalid operator!");
            return;
        }

        System.out.println(num1 + " " + operator + " " + num2 + " = " + result);
    }
}