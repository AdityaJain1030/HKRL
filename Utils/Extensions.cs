using System;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

namespace HKRL.Utils
{
	public static class ColliderExtensions
	{
		/// <summary>
		/// Projects a BoxCollider2D to a list of coordinates on the screen.
		/// </summary>
		/// <param name="box">The BoxCollider2D to convert</param>
		/// <param name="camera">The camera to use for the projection</param>
		public static List<Vector2> ToScreenCoordinates(this BoxCollider2D box, Camera camera)
		{
			Vector2 halfSize = box.size / 2f;
			Vector2 topLeft = new(-halfSize.x, halfSize.y);
			Vector2 topRight = halfSize;
			Vector2 bottomRight = new(halfSize.x, -halfSize.y);
			Vector2 bottomLeft = -halfSize;

			List<Vector2> boxPoints = new List<Vector2>
			{
				topLeft, topRight, bottomRight, bottomLeft
			};

			List<Vector2> projected = boxPoints.Select(point => camera.LocalToScreenPoint(box.transform.TransformPoint(point + box.offset))).ToList();

			return projected;
		}

		/// <summary>
		/// Projects a CircleCollider2D to a center point and radius on the screen.
		/// </summary>
		/// <param name="circle">The CircleCollider2D to convert</param>
		/// <param name="camera">The camera to use for the projection</param>
		public static (Vector2, int) ToScreenCoordinates(this CircleCollider2D circle, Camera camera)
		{

			Vector2 center = camera.LocalToScreenPoint(circle.transform.TransformPoint(circle.offset + Vector2.zero));
			Vector2 right = camera.LocalToScreenPoint(circle.transform.TransformPoint(circle.offset + Vector2.right * circle.radius));
			int radius = (int)Math.Round(Vector2.Distance(center, right));

			return (center, radius);
		}

		/// <summary>
		/// Projects a PolygonCollider2D to a list of coordinates on the screen.
		/// </summary>
		/// <param name="polygon">The PolygonCollider2D to convert</param>
		/// <param name="camera">The camera to use for the projection</param>
		/// <returns>A List of closed shapes represented by a list of points</returns>
		public static List<List<Vector2>> ToScreenCoordinates(this PolygonCollider2D polygon, Camera camera)
		{
			List<List<Vector2>> projection = new();
			for (int i = 0; i < polygon.pathCount; i++)
			{
				List<Vector2> polygonPoints = new(polygon.GetPath(i));
				List<Vector2> projected = polygonPoints.Select(point => camera.LocalToScreenPoint(polygon.transform.TransformPoint(point + polygon.offset))).ToList();
				projection.Add(projected);
			}
			return projection;
		}

		/// <summary>
		/// Projects an EdgeCollider2D to a list of coordinates on the screen.
		/// </summary>
		/// <param name="edge">The EdgeCollider2D to convert</param>
		/// <param name="camera">The camera to use for the projection</param>
		public static List<Vector2> ToScreenCoordinates(this EdgeCollider2D edge, Camera camera)
		{
			List<Vector2> edgePoints = new(edge.points);
			List<Vector2> projected = edgePoints.Select(point => camera.LocalToScreenPoint(edge.transform.TransformPoint(point + edge.offset))).ToList();
			return projected;
		}
	}

	public static class CameraExtensions
	{
		public static Vector2 LocalToScreenPoint(this Camera camera, Vector2 worldPoint)
		{
			Vector2 result = camera.WorldToScreenPoint((Vector2)worldPoint);
			return new Vector2((int)Math.Round(result.x), (int)Math.Round(Screen.height - result.y));
		}
	}

	public static class MathExtensions
	{
		/// <summary>
		/// An optimized method using an array as buffer instead of 
		/// string concatenation. This is faster for return values having 
		/// a length > 1.
		/// </summary>
		public static string IntToBaseThreeString(this int value, int max_digits)
		{
			int i = max_digits;
			char[] baseChars = "012".ToCharArray();
			char[] buffer = new char[i];
			int targetBase = baseChars.Length;

			do
			{
				buffer[--i] = baseChars[value % targetBase];
				value = value / targetBase;
			}
			while (value > 0);

			char[] result = new char[4 - i];
			Array.Copy(buffer, i, result, 0, 4 - i);

			return new string('0', max_digits - result.Length) + new string(result);
		}
	}
}