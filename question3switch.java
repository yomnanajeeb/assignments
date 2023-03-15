import java.util.Scanner;

public class question3switch {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.println("Enter the radius of the circle: ");
        double radius, area;
        radius = scanner.nextDouble();

        switch (Double.compare(radius, 0)) {
            case 1:
                area = Math.PI * radius * radius;
                System.out.println("The area of the circle is: " + area);
                break;
            case 0:
                System.out.println("The radius of the circle cannot be zero!");
                break;
            case -1:
                System.out.println("The radius of the circle cannot be negative!");
                break;
            default:
                System.out.println("Invalid input!");

        }
    }
}