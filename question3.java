
import java.util.Scanner;

public class question3 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.println("Enter the radius of the circle: ");
        double radius = scanner.nextDouble();
        if (radius > 0) {
            double area = Math.PI * radius * radius;
            System.out.println("The area of the circle is: " + area);
        } else if (radius == 0) {
            System.out.println("The radius of the circle cannot be zero!");
        } else {
            System.out.println("The radius of the circle cannot be negative!");
        }
    }
}