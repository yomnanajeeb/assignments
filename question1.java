import java.util.Scanner;
public class question1 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner (System.in);
        System.out.print("Enter the radius of the circle: ");
        double radius = scanner.nextDouble();
        scanner.close();
        double area = Math.PI * radius * radius;
        int roundedArea = (int) Math.round(area);
        System.out.println("The area of the circle is: " + roundedArea);
    }
}
