using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace HKRL.Utils
{
	///
	/// <summary>
	///  A struct that represents a greyscale image.
	///	 The image is stored as a 1D array of bytes, with each byte representing the greyscale
	///	 value of a pixel. The pixels are stored in row-major order, with the first row
	///	 stored first, and the last row stored last.
	/// </summary>
	public struct Image
	{
		/// The width of the Image
		public readonly int width;
		/// The height of the Image
		public readonly int height;
		public readonly byte[] pixels;

		public Image(int width, int height)
		{
			this.width = width;
			this.height = height;
			this.pixels = new byte[width * height];
		}

		public byte this[int x, int y]
		{
			get => pixels[x * height + y];
			private set
			{
				if (x < 0 || x >= width || y < 0 || y >= height)
				{
					return;
				}
				pixels[x * height + y] = value;
			}
		}

		///<summary>
		/// sets all pixels to 0
		///</summary>
		public Image Reset()
		{
			for (int i = 0; i < pixels.Length; i++)
			{
				pixels[i] = 0;
			}
			return this;
		}

		///<summary>
		/// Resizes the Image using nearest neighbor interpolation
		///</summary>
		///<param name="newWidth">The new width of the Image</param>
		///<param name="newHeight">The new height of the Image</param>
		public Image ResizeNearestNeighbor(int newWidth, int newHeight)
		{
			Image newImage = new Image(newWidth, newHeight);
			int x_ratio = (int)((this.width << 16) / newWidth) + 1;
			int y_ratio = (int)((this.height << 16) / newHeight) + 1;

			int x2, y2;
			for (int i = 0; i < newHeight; i++)
			{
				for (int j = 0; j < newWidth; j++)
				{
					x2 = ((j * x_ratio) >> 16);
					y2 = ((i * y_ratio) >> 16);
					newImage[j, i] = this[x2, y2];
				}
			}
			return newImage;
		}

		///<summary>
		/// Rasterizes a circle to the Image
		/// Uses a crude circle algorithm cuz I was too lazy to figure out scan lines.
		///</summary>
		///<param name="x">x coord of center</param>
		///<param name="y">y coord of center</param>
		///<param name="radius">The radius of the circle</param>
		///<param name="greyscale">The greyscale value of the circle (0-255)</param>
		public Image DrawCircle(int x, int y, int radius, byte greyscale)
		{
			for (int i = -radius; i < radius; i++)
			{
				int height = (int)Math.Sqrt(radius * radius - i * i);

				for (int j = -height; j < height; j++)
				{
					this[x + i, y + j] = greyscale;
				}
			}

			return this;
		}

		///<summary>
		/// Rasterizes a rectangle to the Image
		///</summary>
		///<param name="x">x coord of upper left corner</param>
		///<param name="y">y coord of upper left corner</param>
		///<param name="width">The width of the rectangle</param>
		///<param name="height">The height of the rectangle</param>
		///<param name="greyscale">The greyscale value of the rectangle (0-255)</param>
		public Image DrawRectangle(int x, int y, int width, int height, byte greyscale)
		{
			for (int i = 0; i < width; i++)
			{
				for (int j = 0; j < height; j++)
				{
					this[x + i, y + j] = greyscale;
				}
			}
			return this;
		}

		///<summary>
		/// Rasterizes a polygon to the GameObservation
		/// Uses the Efficient Fill Algorithm detailed <a href="http://alienryderflex.com/polygon_fill">here</a>.
		///</summary>
		///<param name="vertecies">A list of the polygon vertexes</param>
		///<param name="greyscale">The greyscale value of the polygon</param>
		public Image DrawPolygon(List<Vector2> vertecies, byte greyscale)
		{
			int SIZE = vertecies.Count;
			int IMAGE_TOP = 0;
			int IMAGE_BOT = height;
			int IMAGE_LEFT = 0;
			int IMAGE_RIGHT = width;

			int[] nodeX = new int[100];
			int
				i,
				j,
				nodes,
				pixelX,
				pixelY,
				swap;

			for (pixelY = IMAGE_TOP; pixelY < IMAGE_BOT; pixelY++)
			{
				nodes = 0;
				j = SIZE - 1;
				for (i = 0; i < SIZE; i++)
				{
					if (
						vertecies[i].y < pixelY && vertecies[j].y >= pixelY ||
						vertecies[j].y < pixelY && vertecies[i].y >= pixelY
					)
					{
						nodeX[nodes++] =
							(
							int
							)(vertecies[i].x +
							(pixelY - vertecies[i].y) /
							(vertecies[j].y - vertecies[i].y) *
							(vertecies[j].x - vertecies[i].x));
					}
					j = i;
				}
				i = 0;
				while (i < nodes - 1)
				{
					if (nodeX[i] > nodeX[i + 1])
					{
						swap = nodeX[i];
						nodeX[i] = nodeX[i + 1];
						nodeX[i + 1] = swap;
						if (i != 0)
						{
							i--;
						}
					}
					else
					{
						i++;
					}
				}
				for (i = 0; i < nodes; i += 2)
				{
					if (nodeX[i] >= IMAGE_RIGHT)
					{
						break;
					}
					if (nodeX[i + 1] > IMAGE_LEFT)
					{
						if (nodeX[i] < IMAGE_LEFT)
						{
							nodeX[i] = IMAGE_LEFT;
						}
						if (nodeX[i + 1] > IMAGE_RIGHT)
						{
							nodeX[i + 1] = IMAGE_RIGHT;
						}
						for (pixelX = nodeX[i]; pixelX < nodeX[i + 1]; pixelX++)
						{
							this[pixelX, ((IMAGE_BOT - 1) - pixelY)] = greyscale;
						}
					}
				}
			}
			return this;
		}

		///<summary>
		/// Stacks the Image on top of another Image with a given multiplier. The image being stacked will be  multiplied by the multiplier before being stacked.
		///</summary>
		///<param name="image">The Image to stack on top of this Image</param>
		///<param name="multiplier">The multiplier to apply to the Image being stacked</param>

		public Image AddImage(Image image, byte multiplier)
		{
			if (image.width != width || image.height != height)
			{
				throw new ArgumentException("Image dimensions must match");
			}

			for (int i = 0; i < width; i++)
			{
				for (int j = 0; j < height; j++)
				{
					this[i, j] = (byte)Math.Min(255, this[i, j] + image[i, j] * multiplier);
				}
			}

			return this;
		}

		// public Image LayerImage(Image image, )

		///<summary>
		/// Resizes the Image using nearest neighbor interpolation
		///</summary>
		///<param name="newSize">The new size of the Image</param>
		public Image ResizeNearestNeighbor(Vector2 newSize) => ResizeNearestNeighbor((int)newSize.x, (int)newSize.y);
		///<summary>
		/// Draws a circle to the Image
		/// Uses a crude circle algorithm cuz I was too lazy to figure out scan lines.
		///</summary>
		///<param name="center">The center of the circle</param>
		///<param name="radius">The radius of the circle</param>
		///<param name="greyscale">The greyscale value of the circle (0-255)</param>
		public Image DrawCircle(Vector2 center, int radius, byte greyscale) => DrawCircle((int)center.x, (int)center.y, radius, greyscale);
		///<summary>
		/// Rasterizes a rectangle to the Image
		///</summary>
		///<param name="upperLeft">The upper left corner of the rectangle</param>
		///<param name="width">The width of the rectangle</param>
		///<param name="height">The height of the rectangle</param>
		///<param name="greyscale">The greyscale value of the rectangle (0-255)</param>
		public Image DrawRectangle(Vector2 upperLeft, int width, int height, byte greyscale) => DrawRectangle((int)upperLeft.x, (int)upperLeft.y, width, height, greyscale);
		///<summary>
		/// Rasterizes a rectangle to the Image
		///</summary>
		///<param name="upperLeft">The upper left corner of the rectangle</param>
		///<param name="size">The size of the rectangle</param>
		///<param name="greyscale">The greyscale value of the rectangle (0-255)</param>
		public Image DrawRectangle(Vector2 upperLeft, Vector2 size, byte greyscale) => DrawRectangle((int)upperLeft.x, (int)upperLeft.y, (int)size.x, (int)size.y, greyscale);

		///<summary>
		/// Converts the binary image to an int array
		///</summary>
		///<returns>An int array from the byte array</returns>
		public int[] ToIntArray()
		{
			return pixels.Select(x => (int)x).ToArray();
		}
	}
}