import java.util.Scanner;

public class question2switch {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.println("please enter number");
        int num1, num2;
        char operator;
        double result =0;

        System.out.println("Enter first number: ");
        num1 = scanner.nextInt();

        System.out.println("Enter second number: ");
        num2 = scanner.nextInt();

        System.out.println("Enter an operator (+, -, *, /): ");
        operator = scanner.next().charAt(0);

        switch (operator) {
            case '+':
                result = num1 + num2;
                break;
            case '-':
                result = num1 - num2;
                break;
            case '*':
                result = num1 * num2;
                break;
            case '/':
                result = num1 / num2;
                break;
            default:
                System.out.println("Invalid operator!");
        }
        System.out.println(num1 + " " + operator + " " + num2 + " = " + result);
    }
}