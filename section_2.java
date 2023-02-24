import java.util.Scanner;

public class section_2 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.println("enter the first number");
        int x = scanner.nextInt();
        System.out.println("enter the second number");
        int y = scanner.nextInt();
        System.out.println(x + y);
        System.out.println(x - y);
        System.out.println(x * y);
        System.out.println(x / y);
    }
}